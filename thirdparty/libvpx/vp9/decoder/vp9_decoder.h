/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_DECODER_VP9_DECODER_H_
#define VPX_VP9_DECODER_VP9_DECODER_H_

#include "./vpx_config.h"

#include "vpx/vpx_codec.h"
#include "vpx_dsp/bitreader.h"
#include "vpx_scale/yv12config.h"
#include "vpx_util/vpx_pthread.h"
#include "vpx_util/vpx_thread.h"

#include "vp9/common/vp9_thread_common.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_ppflags.h"
#include "./vp9_job_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EOBS_PER_SB_LOG2 8
#define DQCOEFFS_PER_SB_LOG2 12
#define PARTITIONS_PER_SB 85

typedef enum JobType { PARSE_JOB, RECON_JOB, LPF_JOB } JobType;

typedef struct ThreadData {
  struct VP9Decoder *pbi;
  LFWorkerData *lf_data;
  VP9LfSync *lf_sync;
} ThreadData;

typedef struct TileBuffer {
  const uint8_t *data;
  size_t size;
  int col;  // only used with multi-threaded decoding
} TileBuffer;

typedef struct TileWorkerData {
  const uint8_t *data_end;
  int buf_start, buf_end;  // pbi->tile_buffers to decode, inclusive
  vpx_reader bit_reader;
  FRAME_COUNTS counts;
  LFWorkerData *lf_data;
  VP9LfSync *lf_sync;
  DECLARE_ALIGNED(16, MACROBLOCKD, xd);
  /* dqcoeff are shared by all the planes. So planes must be decoded serially */
  DECLARE_ALIGNED(32, tran_low_t, dqcoeff[32 * 32]);
  DECLARE_ALIGNED(16, uint16_t, extend_and_predict_buf[80 * 2 * 80 * 2]);
  struct vpx_internal_error_info error_info;
} TileWorkerData;

typedef void (*process_block_fn_t)(TileWorkerData *twd,
                                   struct VP9Decoder *const pbi, int mi_row,
                                   int mi_col, BLOCK_SIZE bsize, int bwl,
                                   int bhl);

typedef struct RowMTWorkerData {
  int num_sbs;
  int *eob[MAX_MB_PLANE];
  PARTITION_TYPE *partition;
  tran_low_t *dqcoeff[MAX_MB_PLANE];
  int8_t *recon_map;
  const uint8_t *data_end;
  uint8_t *jobq_buf;
  JobQueueRowMt jobq;
  size_t jobq_size;
  int num_tiles_done;
  int num_jobs;
#if CONFIG_MULTITHREAD
  pthread_mutex_t recon_done_mutex;
  pthread_mutex_t *recon_sync_mutex;
  pthread_cond_t *recon_sync_cond;
#endif
  ThreadData *thread_data;
} RowMTWorkerData;

/* Structure to queue and dequeue row decode jobs */
typedef struct Job {
  int row_num;
  int tile_col;
  JobType job_type;
} Job;

typedef struct VP9Decoder {
  DECLARE_ALIGNED(16, MACROBLOCKD, mb);

  DECLARE_ALIGNED(16, VP9_COMMON, common);

  int ready_for_new_data;

  int refresh_frame_flags;

  // TODO(hkuang): Combine this with cur_buf in macroblockd as they are
  // the same.
  RefCntBuffer *cur_buf;  //  Current decoding frame buffer.

  VPxWorker lf_worker;
  VPxWorker *tile_workers;
  TileWorkerData *tile_worker_data;
  TileBuffer tile_buffers[64];
  int num_tile_workers;
  int total_tiles;

  VP9LfSync lf_row_sync;

  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;

  int max_threads;
  int inv_tile_order;
  int need_resync;   // wait for key/intra-only frame.
  int hold_ref_buf;  // hold the reference buffer.

  int row_mt;
  int lpf_mt_opt;
  RowMTWorkerData *row_mt_worker_data;
} VP9Decoder;

int vp9_receive_compressed_data(struct VP9Decoder *pbi, size_t size,
                                const uint8_t **psource);

int vp9_get_raw_frame(struct VP9Decoder *pbi, YV12_BUFFER_CONFIG *sd,
                      vp9_ppflags_t *flags);

vpx_codec_err_t vp9_copy_reference_dec(struct VP9Decoder *pbi,
                                       VP9_REFFRAME ref_frame_flag,
                                       YV12_BUFFER_CONFIG *sd);

vpx_codec_err_t vp9_set_reference_dec(VP9_COMMON *cm,
                                      VP9_REFFRAME ref_frame_flag,
                                      YV12_BUFFER_CONFIG *sd);

static INLINE uint8_t read_marker(vpx_decrypt_cb decrypt_cb,
                                  void *decrypt_state, const uint8_t *data) {
  if (decrypt_cb) {
    uint8_t marker;
    decrypt_cb(decrypt_state, data, &marker, 1);
    return marker;
  }
  return *data;
}

// This function is exposed for use in tests, as well as the inlined function
// "read_marker".
vpx_codec_err_t vp9_parse_superframe_index(const uint8_t *data, size_t data_sz,
                                           uint32_t sizes[8], int *count,
                                           vpx_decrypt_cb decrypt_cb,
                                           void *decrypt_state);

struct VP9Decoder *vp9_decoder_create(BufferPool *const pool);

void vp9_decoder_remove(struct VP9Decoder *pbi);

void vp9_dec_alloc_row_mt_mem(RowMTWorkerData *row_mt_worker_data,
                              VP9_COMMON *cm, int num_sbs, int max_threads,
                              int num_jobs);
void vp9_dec_free_row_mt_mem(RowMTWorkerData *row_mt_worker_data);

static INLINE void decrease_ref_count(int idx, RefCntBuffer *const frame_bufs,
                                      BufferPool *const pool) {
  if (idx >= 0 && frame_bufs[idx].ref_count > 0) {
    --frame_bufs[idx].ref_count;
    // A worker may only get a free framebuffer index when calling get_free_fb.
    // But the private buffer is not set up until finish decoding header.
    // So any error happens during decoding header, the frame_bufs will not
    // have valid priv buffer.
    if (!frame_bufs[idx].released && frame_bufs[idx].ref_count == 0 &&
        frame_bufs[idx].raw_frame_buffer.priv) {
      pool->release_fb_cb(pool->cb_priv, &frame_bufs[idx].raw_frame_buffer);
      frame_bufs[idx].released = 1;
    }
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_DECODER_VP9_DECODER_H_
