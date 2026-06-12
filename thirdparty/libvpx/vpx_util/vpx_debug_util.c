/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "vpx_util/vpx_debug_util.h"

#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
static int frame_idx_w = 0;
static int frame_idx_r = 0;

void bitstream_queue_set_frame_write(int frame_idx) { frame_idx_w = frame_idx; }

int bitstream_queue_get_frame_write(void) { return frame_idx_w; }

void bitstream_queue_set_frame_read(int frame_idx) { frame_idx_r = frame_idx; }

int bitstream_queue_get_frame_read(void) { return frame_idx_r; }
#endif

#if CONFIG_BITSTREAM_DEBUG
#define QUEUE_MAX_SIZE 2000000
static int result_queue[QUEUE_MAX_SIZE];
static int prob_queue[QUEUE_MAX_SIZE];

static int queue_r = 0;
static int queue_w = 0;
static int queue_prev_w = -1;
static int skip_r = 0;
static int skip_w = 0;
void bitstream_queue_set_skip_write(int skip) { skip_w = skip; }

void bitstream_queue_set_skip_read(int skip) { skip_r = skip; }

void bitstream_queue_record_write(void) { queue_prev_w = queue_w; }

void bitstream_queue_reset_write(void) { queue_w = queue_prev_w; }

int bitstream_queue_get_write(void) { return queue_w; }

int bitstream_queue_get_read(void) { return queue_r; }

void bitstream_queue_pop(int *result, int *prob) {
  if (!skip_r) {
    if (queue_w == queue_r) {
      printf("buffer underflow queue_w %d queue_r %d\n", queue_w, queue_r);
      assert(0);
    }
    *result = result_queue[queue_r];
    *prob = prob_queue[queue_r];
    queue_r = (queue_r + 1) % QUEUE_MAX_SIZE;
  }
}

void bitstream_queue_push(int result, const int prob) {
  if (!skip_w) {
    result_queue[queue_w] = result;
    prob_queue[queue_w] = prob;
    queue_w = (queue_w + 1) % QUEUE_MAX_SIZE;
    if (queue_w == queue_r) {
      printf("buffer overflow queue_w %d queue_r %d\n", queue_w, queue_r);
      assert(0);
    }
  }
}
#endif  // CONFIG_BITSTREAM_DEBUG

#if CONFIG_MISMATCH_DEBUG
static int frame_buf_idx_r = 0;
static int frame_buf_idx_w = 0;
#define MAX_FRAME_BUF_NUM 20
#define MAX_FRAME_STRIDE 1920
#define MAX_FRAME_HEIGHT 1080
static uint16_t
    frame_pre[MAX_FRAME_BUF_NUM][3]
             [MAX_FRAME_STRIDE * MAX_FRAME_HEIGHT];  // prediction only
static uint16_t
    frame_tx[MAX_FRAME_BUF_NUM][3]
            [MAX_FRAME_STRIDE * MAX_FRAME_HEIGHT];  // prediction + txfm
static int frame_stride = MAX_FRAME_STRIDE;
static int frame_height = MAX_FRAME_HEIGHT;
static int frame_size = MAX_FRAME_STRIDE * MAX_FRAME_HEIGHT;
void mismatch_move_frame_idx_w(void) {
  frame_buf_idx_w = (frame_buf_idx_w + 1) % MAX_FRAME_BUF_NUM;
  if (frame_buf_idx_w == frame_buf_idx_r) {
    printf("frame_buf overflow\n");
    assert(0);
  }
}

void mismatch_reset_frame(int num_planes) {
  int plane;
  for (plane = 0; plane < num_planes; ++plane) {
    memset(frame_pre[frame_buf_idx_w][plane], 0,
           sizeof(frame_pre[frame_buf_idx_w][plane][0]) * frame_size);
    memset(frame_tx[frame_buf_idx_w][plane], 0,
           sizeof(frame_tx[frame_buf_idx_w][plane][0]) * frame_size);
  }
}

void mismatch_move_frame_idx_r(void) {
  if (frame_buf_idx_w == frame_buf_idx_r) {
    printf("frame_buf underflow\n");
    assert(0);
  }
  frame_buf_idx_r = (frame_buf_idx_r + 1) % MAX_FRAME_BUF_NUM;
}

void mismatch_record_block_pre(const uint8_t *src, int src_stride, int plane,
                               int pixel_c, int pixel_r, int blk_w, int blk_h,
                               int highbd) {
  const uint16_t *src16 = highbd ? CONVERT_TO_SHORTPTR(src) : NULL;
  int r, c;

  if (pixel_c + blk_w >= frame_stride || pixel_r + blk_h >= frame_height) {
    printf("frame_buf undersized\n");
    assert(0);
  }

  for (r = 0; r < blk_h; ++r) {
    for (c = 0; c < blk_w; ++c) {
      frame_pre[frame_buf_idx_w][plane]
               [(r + pixel_r) * frame_stride + c + pixel_c] =
                   src16 ? src16[r * src_stride + c] : src[r * src_stride + c];
    }
  }
#if 0
  {
    int ref_frame_idx = 3;
    int ref_plane = 1;
    int ref_pixel_c = 162;
    int ref_pixel_r = 16;
    if (frame_idx_w == ref_frame_idx && plane == ref_plane &&
        ref_pixel_c >= pixel_c && ref_pixel_c < pixel_c + blk_w &&
        ref_pixel_r >= pixel_r && ref_pixel_r < pixel_r + blk_h) {
      printf(
          "\nrecord_block_pre frame_idx %d plane %d pixel_c %d pixel_r %d blk_w"
          " %d blk_h %d\n",
          frame_idx_w, plane, pixel_c, pixel_r, blk_w, blk_h);
    }
  }
#endif
}
void mismatch_record_block_tx(const uint8_t *src, int src_stride, int plane,
                              int pixel_c, int pixel_r, int blk_w, int blk_h,
                              int highbd) {
  const uint16_t *src16 = highbd ? CONVERT_TO_SHORTPTR(src) : NULL;
  int r, c;
  if (pixel_c + blk_w >= frame_stride || pixel_r + blk_h >= frame_height) {
    printf("frame_buf undersized\n");
    assert(0);
  }

  for (r = 0; r < blk_h; ++r) {
    for (c = 0; c < blk_w; ++c) {
      frame_tx[frame_buf_idx_w][plane]
              [(r + pixel_r) * frame_stride + c + pixel_c] =
                  src16 ? src16[r * src_stride + c] : src[r * src_stride + c];
    }
  }
#if 0
  {
    int ref_frame_idx = 3;
    int ref_plane = 1;
    int ref_pixel_c = 162;
    int ref_pixel_r = 16;
    if (frame_idx_w == ref_frame_idx && plane == ref_plane &&
        ref_pixel_c >= pixel_c && ref_pixel_c < pixel_c + blk_w &&
        ref_pixel_r >= pixel_r && ref_pixel_r < pixel_r + blk_h) {
      printf(
          "\nrecord_block_tx frame_idx %d plane %d pixel_c %d pixel_r %d blk_w "
          "%d blk_h %d\n",
          frame_idx_w, plane, pixel_c, pixel_r, blk_w, blk_h);
    }
  }
#endif
}
void mismatch_check_block_pre(const uint8_t *src, int src_stride, int plane,
                              int pixel_c, int pixel_r, int blk_w, int blk_h,
                              int highbd) {
  const uint16_t *src16 = highbd ? CONVERT_TO_SHORTPTR(src) : NULL;
  int mismatch = 0;
  int r, c;
  if (pixel_c + blk_w >= frame_stride || pixel_r + blk_h >= frame_height) {
    printf("frame_buf undersized\n");
    assert(0);
  }

  for (r = 0; r < blk_h; ++r) {
    for (c = 0; c < blk_w; ++c) {
      if (frame_pre[frame_buf_idx_r][plane]
                   [(r + pixel_r) * frame_stride + c + pixel_c] !=
          (uint16_t)(src16 ? src16[r * src_stride + c]
                           : src[r * src_stride + c])) {
        mismatch = 1;
      }
    }
  }
  if (mismatch) {
    int rr, cc;
    printf(
        "\ncheck_block_pre failed frame_idx %d plane %d "
        "pixel_c %d pixel_r "
        "%d blk_w %d blk_h %d\n",
        frame_idx_r, plane, pixel_c, pixel_r, blk_w, blk_h);
    printf("enc\n");
    for (rr = 0; rr < blk_h; ++rr) {
      for (cc = 0; cc < blk_w; ++cc) {
        printf("%d ", frame_pre[frame_buf_idx_r][plane]
                               [(rr + pixel_r) * frame_stride + cc + pixel_c]);
      }
      printf("\n");
    }

    printf("dec\n");
    for (rr = 0; rr < blk_h; ++rr) {
      for (cc = 0; cc < blk_w; ++cc) {
        printf("%d ",
               src16 ? src16[rr * src_stride + cc] : src[rr * src_stride + cc]);
      }
      printf("\n");
    }
    assert(0);
  }
}
void mismatch_check_block_tx(const uint8_t *src, int src_stride, int plane,
                             int pixel_c, int pixel_r, int blk_w, int blk_h,
                             int highbd) {
  const uint16_t *src16 = highbd ? CONVERT_TO_SHORTPTR(src) : NULL;
  int mismatch = 0;
  int r, c;
  if (pixel_c + blk_w >= frame_stride || pixel_r + blk_h >= frame_height) {
    printf("frame_buf undersized\n");
    assert(0);
  }

  for (r = 0; r < blk_h; ++r) {
    for (c = 0; c < blk_w; ++c) {
      if (frame_tx[frame_buf_idx_r][plane]
                  [(r + pixel_r) * frame_stride + c + pixel_c] !=
          (uint16_t)(src16 ? src16[r * src_stride + c]
                           : src[r * src_stride + c])) {
        mismatch = 1;
      }
    }
  }
  if (mismatch) {
    int rr, cc;
    printf(
        "\ncheck_block_tx failed frame_idx %d plane %d pixel_c "
        "%d pixel_r "
        "%d blk_w %d blk_h %d\n",
        frame_idx_r, plane, pixel_c, pixel_r, blk_w, blk_h);
    printf("enc\n");
    for (rr = 0; rr < blk_h; ++rr) {
      for (cc = 0; cc < blk_w; ++cc) {
        printf("%d ", frame_tx[frame_buf_idx_r][plane]
                              [(rr + pixel_r) * frame_stride + cc + pixel_c]);
      }
      printf("\n");
    }

    printf("dec\n");
    for (rr = 0; rr < blk_h; ++rr) {
      for (cc = 0; cc < blk_w; ++cc) {
        printf("%d ",
               src16 ? src16[rr * src_stride + cc] : src[rr * src_stride + cc]);
      }
      printf("\n");
    }
    assert(0);
  }
}
#endif  // CONFIG_MISMATCH_DEBUG
