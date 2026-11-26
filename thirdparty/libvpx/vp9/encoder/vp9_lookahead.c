/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"

#include "vp9/common/vp9_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_extend.h"
#include "vp9/encoder/vp9_lookahead.h"

/* Return the buffer at the given absolute index and increment the index */
static struct lookahead_entry *pop(struct lookahead_ctx *ctx, int *idx) {
  int index = *idx;
  struct lookahead_entry *buf = ctx->buf + index;

  assert(index < ctx->max_sz);
  if (++index >= ctx->max_sz) index -= ctx->max_sz;
  *idx = index;
  return buf;
}

void vp9_lookahead_destroy(struct lookahead_ctx *ctx) {
  if (ctx) {
    if (ctx->buf) {
      int i;

      for (i = 0; i < ctx->max_sz; i++) vpx_free_frame_buffer(&ctx->buf[i].img);
      free(ctx->buf);
    }
    free(ctx);
  }
}

struct lookahead_ctx *vp9_lookahead_init(unsigned int width,
                                         unsigned int height,
                                         unsigned int subsampling_x,
                                         unsigned int subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                         int use_highbitdepth,
#endif
                                         unsigned int depth) {
  struct lookahead_ctx *ctx = NULL;

  // Clamp the lookahead queue depth
  depth = clamp(depth, 1, MAX_LAG_BUFFERS);

  // Allocate memory to keep previous source frames available.
  depth += MAX_PRE_FRAMES;

  // Allocate the lookahead structures
  ctx = calloc(1, sizeof(*ctx));
  if (ctx) {
    const int legacy_byte_alignment = 0;
    unsigned int i;
    ctx->max_sz = depth;
    ctx->buf = calloc(depth, sizeof(*ctx->buf));
    ctx->next_show_idx = 0;
    if (!ctx->buf) goto bail;
    for (i = 0; i < depth; i++)
      if (vpx_alloc_frame_buffer(
              &ctx->buf[i].img, width, height, subsampling_x, subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
              use_highbitdepth,
#endif
              VP9_ENC_BORDER_IN_PIXELS, legacy_byte_alignment))
        goto bail;
  }
  return ctx;
bail:
  vp9_lookahead_destroy(ctx);
  return NULL;
}

int vp9_lookahead_full(const struct lookahead_ctx *ctx) {
  return ctx->sz + 1 + MAX_PRE_FRAMES > ctx->max_sz;
}

int vp9_lookahead_next_show_idx(const struct lookahead_ctx *ctx) {
  return ctx->next_show_idx;
}

int vp9_lookahead_push(struct lookahead_ctx *ctx, YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end, int use_highbitdepth,
                       vpx_enc_frame_flags_t flags) {
  struct lookahead_entry *buf;
  int width = src->y_crop_width;
  int height = src->y_crop_height;
  int uv_width = src->uv_crop_width;
  int uv_height = src->uv_crop_height;
  int subsampling_x = src->subsampling_x;
  int subsampling_y = src->subsampling_y;
  int larger_dimensions, new_dimensions;
#if !CONFIG_VP9_HIGHBITDEPTH
  (void)use_highbitdepth;
  assert(use_highbitdepth == 0);
#endif

  if (vp9_lookahead_full(ctx)) return 1;
  ctx->sz++;
  buf = pop(ctx, &ctx->write_idx);

  new_dimensions = width != buf->img.y_crop_width ||
                   height != buf->img.y_crop_height ||
                   uv_width != buf->img.uv_crop_width ||
                   uv_height != buf->img.uv_crop_height;
  larger_dimensions =
      width > buf->img.y_crop_width || height > buf->img.y_crop_height ||
      uv_width > buf->img.uv_crop_width || uv_height > buf->img.uv_crop_height;
  assert(!larger_dimensions || new_dimensions);

  if (larger_dimensions) {
    YV12_BUFFER_CONFIG new_img;
    memset(&new_img, 0, sizeof(new_img));
    if (vpx_alloc_frame_buffer(&new_img, width, height, subsampling_x,
                               subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                               use_highbitdepth,
#endif
                               VP9_ENC_BORDER_IN_PIXELS, 0))
      return 1;
    vpx_free_frame_buffer(&buf->img);
    buf->img = new_img;
  } else if (new_dimensions) {
    int aligned_width = ALIGN_POWER_OF_TWO(width, 3);
    buf->img.y_width = src->y_width;
    buf->img.y_height = src->y_height;
    buf->img.uv_width = src->uv_width;
    buf->img.uv_height = src->uv_height;
    buf->img.y_crop_width = src->y_crop_width;
    buf->img.y_crop_height = src->y_crop_height;
    buf->img.uv_crop_width = src->uv_crop_width;
    buf->img.uv_crop_height = src->uv_crop_height;
    buf->img.subsampling_x = src->subsampling_x;
    buf->img.subsampling_y = src->subsampling_y;
    // Here the new width (src->y_crop_width) is <= the previous width
    // (since otherwise it would enter the "larger_dimensions" code), so
    // it is safe here to update the stride.
    // The stride setting is taken from vpx_alloc_frame_buffer().
    buf->img.y_stride =
        ALIGN_POWER_OF_TWO((aligned_width + 2 * buf->img.border), 5);
    buf->img.uv_stride = buf->img.y_stride >> subsampling_x;
  }
  vp9_copy_and_extend_frame(src, &buf->img);

  buf->ts_start = ts_start;
  buf->ts_end = ts_end;
  buf->flags = flags;
  buf->show_idx = ctx->next_show_idx;
  ++ctx->next_show_idx;
  return 0;
}

struct lookahead_entry *vp9_lookahead_pop(struct lookahead_ctx *ctx,
                                          int drain) {
  struct lookahead_entry *buf = NULL;

  if (ctx && ctx->sz && (drain || ctx->sz == ctx->max_sz - MAX_PRE_FRAMES)) {
    buf = pop(ctx, &ctx->read_idx);
    ctx->sz--;
  }
  return buf;
}

struct lookahead_entry *vp9_lookahead_peek(struct lookahead_ctx *ctx,
                                           int index) {
  struct lookahead_entry *buf = NULL;

  if (index >= 0) {
    // Forward peek
    if (index < ctx->sz) {
      index += ctx->read_idx;
      if (index >= ctx->max_sz) index -= ctx->max_sz;
      buf = ctx->buf + index;
    }
  } else if (index < 0) {
    // Backward peek
    if (-index <= MAX_PRE_FRAMES) {
      index += ctx->read_idx;
      if (index < 0) index += ctx->max_sz;
      buf = ctx->buf + index;
    }
  }

  return buf;
}

unsigned int vp9_lookahead_depth(struct lookahead_ctx *ctx) { return ctx->sz; }
