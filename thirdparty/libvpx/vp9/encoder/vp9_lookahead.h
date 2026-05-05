/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_LOOKAHEAD_H_
#define VPX_VP9_ENCODER_VP9_LOOKAHEAD_H_

#include "vpx_scale/yv12config.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LAG_BUFFERS 25

struct lookahead_entry {
  YV12_BUFFER_CONFIG img;
  int64_t ts_start;
  int64_t ts_end;
  int show_idx; /*The show_idx of this frame*/
  vpx_enc_frame_flags_t flags;
};

// The max of past frames we want to keep in the queue.
#define MAX_PRE_FRAMES 1

struct lookahead_ctx {
  int max_sz;        /* Absolute size of the queue */
  int sz;            /* Number of buffers currently in the queue */
  int read_idx;      /* Read index */
  int write_idx;     /* Write index */
  int next_show_idx; /* The show_idx that will be assigned to the next frame
                        being pushed in the queue*/
  struct lookahead_entry *buf; /* Buffer list */
};

/**\brief Initializes the lookahead stage
 *
 * The lookahead stage is a queue of frame buffers on which some analysis
 * may be done when buffers are enqueued.
 */
struct lookahead_ctx *vp9_lookahead_init(unsigned int width,
                                         unsigned int height,
                                         unsigned int subsampling_x,
                                         unsigned int subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                         int use_highbitdepth,
#endif
                                         unsigned int depth);

/**\brief Destroys the lookahead stage
 */
void vp9_lookahead_destroy(struct lookahead_ctx *ctx);

/**\brief Check if lookahead is full
 *
 * \param[in] ctx         Pointer to the lookahead context
 *
 * Return 1 if lookahead is full, otherwise return 0.
 */
int vp9_lookahead_full(const struct lookahead_ctx *ctx);

/**\brief Return the next_show_idx
 *
 * \param[in] ctx         Pointer to the lookahead context
 *
 * Return the show_idx that will be assigned to the next
 * frame pushed by vp9_lookahead_push()
 */
int vp9_lookahead_next_show_idx(const struct lookahead_ctx *ctx);

/**\brief Enqueue a source buffer
 *
 * This function will copy the source image into a new framebuffer with
 * the expected stride/border.
 *
 * \param[in] ctx         Pointer to the lookahead context
 * \param[in] src         Pointer to the image to enqueue
 * \param[in] ts_start    Timestamp for the start of this frame
 * \param[in] ts_end      Timestamp for the end of this frame
 * \param[in] flags       Flags set on this frame
 */
int vp9_lookahead_push(struct lookahead_ctx *ctx, YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end, int use_highbitdepth,
                       vpx_enc_frame_flags_t flags);

/**\brief Get the next source buffer to encode
 *
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] drain     Flag indicating the buffer should be drained
 *                      (return a buffer regardless of the current queue depth)
 *
 * \retval NULL, if drain set and queue is empty
 * \retval NULL, if drain not set and queue not of the configured depth
 */
struct lookahead_entry *vp9_lookahead_pop(struct lookahead_ctx *ctx, int drain);

/**\brief Get a future source buffer to encode
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] index     Index of the frame to be returned, 0 == next frame
 *
 * \retval NULL, if no buffer exists at the specified index
 */
struct lookahead_entry *vp9_lookahead_peek(struct lookahead_ctx *ctx,
                                           int index);

/**\brief Get the number of frames currently in the lookahead queue
 *
 * \param[in] ctx       Pointer to the lookahead context
 */
unsigned int vp9_lookahead_depth(struct lookahead_ctx *ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_LOOKAHEAD_H_
