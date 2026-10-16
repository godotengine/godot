/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VP8_ENCODER_LOOKAHEAD_H_
#define VPX_VP8_ENCODER_LOOKAHEAD_H_
#include "vpx_scale/yv12config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lookahead_entry {
  YV12_BUFFER_CONFIG img;
  int64_t ts_start;
  int64_t ts_end;
  unsigned int flags;
};

struct lookahead_ctx;

/**\brief Initializes the lookahead stage
 *
 * The lookahead stage is a queue of frame buffers on which some analysis
 * may be done when buffers are enqueued.
 *
 *
 */
struct lookahead_ctx *vp8_lookahead_init(unsigned int width,
                                         unsigned int height,
                                         unsigned int depth);

/**\brief Destroys the lookahead stage
 *
 */
void vp8_lookahead_destroy(struct lookahead_ctx *ctx);

/**\brief Enqueue a source buffer
 *
 * This function will copy the source image into a new framebuffer with
 * the expected stride/border.
 *
 * If active_map is non-NULL and there is only one frame in the queue, then copy
 * only active macroblocks.
 *
 * \param[in] ctx         Pointer to the lookahead context
 * \param[in] src         Pointer to the image to enqueue
 * \param[in] ts_start    Timestamp for the start of this frame
 * \param[in] ts_end      Timestamp for the end of this frame
 * \param[in] flags       Flags set on this frame
 * \param[in] active_map  Map that specifies which macroblock is active
 */
int vp8_lookahead_push(struct lookahead_ctx *ctx, YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end, unsigned int flags,
                       unsigned char *active_map);

/**\brief Get the next source buffer to encode
 *
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] drain     Flag indicating the buffer should be drained
 *                      (return a buffer regardless of the current queue depth)
 *
 * \retval NULL, if drain set and queue is empty
 * \retval NULL, if drain not set and queue not of the configured depth
 *
 */
struct lookahead_entry *vp8_lookahead_pop(struct lookahead_ctx *ctx, int drain);

#define PEEK_FORWARD 1
#define PEEK_BACKWARD (-1)
/**\brief Get a future source buffer to encode
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] index     Index of the frame to be returned, 0 == next frame
 *
 * \retval NULL, if no buffer exists at the specified index
 *
 */
struct lookahead_entry *vp8_lookahead_peek(struct lookahead_ctx *ctx,
                                           unsigned int index, int direction);

/**\brief Get the number of frames currently in the lookahead queue
 *
 * \param[in] ctx       Pointer to the lookahead context
 */
unsigned int vp8_lookahead_depth(struct lookahead_ctx *ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_LOOKAHEAD_H_
