// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//  AnimEncoder implementation.
//

#include <assert.h>
#include <limits.h>
#include <math.h>    // for pow()
#include <stdio.h>
#include <stdlib.h>  // for abs()
#include <string.h>

#include "src/mux/animi.h"
#include "src/utils/utils.h"
#include "src/webp/decode.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"
#include "src/webp/mux.h"
#include "src/webp/mux_types.h"
#include "src/webp/types.h"

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

#define ERROR_STR_MAX_LENGTH 100

//------------------------------------------------------------------------------
// Internal structs.

// Stores frame rectangle dimensions.
typedef struct {
  int x_offset, y_offset, width, height;
} FrameRectangle;

// Used to store two candidates of encoded data for an animation frame. One of
// the two will be chosen later.
typedef struct {
  WebPMuxFrameInfo sub_frame;  // Encoded frame rectangle.
  WebPMuxFrameInfo key_frame;  // Encoded frame if it is a key-frame.
  int is_key_frame;            // True if 'key_frame' has been chosen.
} EncodedFrame;

struct WebPAnimEncoder {
  const int canvas_width;                  // Canvas width.
  const int canvas_height;                 // Canvas height.
  const WebPAnimEncoderOptions options;    // Global encoding options.

  FrameRectangle prev_rect;           // Previous WebP frame rectangle.
  WebPConfig last_config;             // Cached in case a re-encode is needed.
  WebPConfig last_config_reversed;    // If 'last_config' uses lossless, then
                                      // this config uses lossy and vice versa;
                                      // only valid if 'options.allow_mixed'
                                      // is true.

  WebPPicture* curr_canvas;           // Only pointer; we don't own memory.

  // Canvas buffers.
  WebPPicture curr_canvas_copy;       // Possibly modified current canvas.
  int curr_canvas_copy_modified;      // True if pixels in 'curr_canvas_copy'
                                      // differ from those in 'curr_canvas'.

  WebPPicture prev_canvas;            // Previous canvas.
  WebPPicture prev_canvas_disposed;   // Previous canvas disposed to background.

  // Encoded data.
  EncodedFrame* encoded_frames;       // Array of encoded frames.
  size_t size;              // Number of allocated frames.
  size_t start;             // Frame start index.
  size_t count;             // Number of valid frames.
  size_t flush_count;       // If >0, 'flush_count' frames starting from
                            // 'start' are ready to be added to mux.

  // key-frame related.
  int64_t best_delta;       // min(canvas size - frame size) over the frames.
                            // Can be negative in certain cases due to
                            // transparent pixels in a frame.
  int keyframe;             // Index of selected key-frame relative to 'start'.
  int count_since_key_frame;      // Frames seen since the last key-frame.

  int first_timestamp;            // Timestamp of the first frame.
  int prev_timestamp;             // Timestamp of the last added frame.
  int prev_candidate_undecided;   // True if it's not yet decided if previous
                                  // frame would be a sub-frame or a key-frame.

  // Misc.
  int is_first_frame;   // True if first frame is yet to be added/being added.
  int got_null_frame;   // True if WebPAnimEncoderAdd() has already been called
                        // with a NULL frame.

  size_t in_frame_count;   // Number of input frames processed so far.
  size_t out_frame_count;  // Number of frames added to mux so far. This may be
                           // different from 'in_frame_count' due to merging.

  WebPMux* mux;         // Muxer to assemble the WebP bitstream.
  char error_str[ERROR_STR_MAX_LENGTH];  // Error string. Empty if no error.
};

// -----------------------------------------------------------------------------
// Life of WebPAnimEncoder object.

#define DELTA_INFINITY      (1ULL << 32)
#define KEYFRAME_NONE       (-1)

// Reset the counters in the WebPAnimEncoder.
static void ResetCounters(WebPAnimEncoder* const enc) {
  enc->start = 0;
  enc->count = 0;
  enc->flush_count = 0;
  enc->best_delta = DELTA_INFINITY;
  enc->keyframe = KEYFRAME_NONE;
}

static void DisableKeyframes(WebPAnimEncoderOptions* const enc_options) {
  enc_options->kmax = INT_MAX;
  enc_options->kmin = enc_options->kmax - 1;
}

#define MAX_CACHED_FRAMES 30

static void SanitizeEncoderOptions(WebPAnimEncoderOptions* const enc_options) {
  int print_warning = enc_options->verbose;

  if (enc_options->minimize_size) {
    DisableKeyframes(enc_options);
  }

  if (enc_options->kmax == 1) {  // All frames will be key-frames.
    enc_options->kmin = 0;
    enc_options->kmax = 0;
    return;
  } else if (enc_options->kmax <= 0) {
    DisableKeyframes(enc_options);
    print_warning = 0;
  }

  if (enc_options->kmin >= enc_options->kmax) {
    enc_options->kmin = enc_options->kmax - 1;
    if (print_warning) {
      fprintf(stderr, "WARNING: Setting kmin = %d, so that kmin < kmax.\n",
              enc_options->kmin);
    }
  } else {
    const int kmin_limit = enc_options->kmax / 2 + 1;
    if (enc_options->kmin < kmin_limit && kmin_limit < enc_options->kmax) {
      // This ensures that enc.keyframe + kmin >= kmax is always true. So, we
      // can flush all the frames in the 'count_since_key_frame == kmax' case.
      enc_options->kmin = kmin_limit;
      if (print_warning) {
        fprintf(stderr,
                "WARNING: Setting kmin = %d, so that kmin >= kmax / 2 + 1.\n",
                enc_options->kmin);
      }
    }
  }
  // Limit the max number of frames that are allocated.
  if (enc_options->kmax - enc_options->kmin > MAX_CACHED_FRAMES) {
    enc_options->kmin = enc_options->kmax - MAX_CACHED_FRAMES;
    if (print_warning) {
      fprintf(stderr,
              "WARNING: Setting kmin = %d, so that kmax - kmin <= %d.\n",
              enc_options->kmin, MAX_CACHED_FRAMES);
    }
  }
  assert(enc_options->kmin < enc_options->kmax);
}

#undef MAX_CACHED_FRAMES

static void DefaultEncoderOptions(WebPAnimEncoderOptions* const enc_options) {
  enc_options->anim_params.loop_count = 0;
  enc_options->anim_params.bgcolor = 0xffffffff;  // White.
  enc_options->minimize_size = 0;
  DisableKeyframes(enc_options);
  enc_options->allow_mixed = 0;
  enc_options->verbose = 0;
}

int WebPAnimEncoderOptionsInitInternal(WebPAnimEncoderOptions* enc_options,
                                       int abi_version) {
  if (enc_options == NULL ||
      WEBP_ABI_IS_INCOMPATIBLE(abi_version, WEBP_MUX_ABI_VERSION)) {
    return 0;
  }
  DefaultEncoderOptions(enc_options);
  return 1;
}

// This value is used to match a later call to WebPReplaceTransparentPixels(),
// making it a no-op for lossless (see WebPEncode()).
#define TRANSPARENT_COLOR   0x00000000

static void ClearRectangle(WebPPicture* const picture,
                           int left, int top, int width, int height) {
  int j;
  for (j = top; j < top + height; ++j) {
    uint32_t* const dst = picture->argb + j * picture->argb_stride;
    int i;
    for (i = left; i < left + width; ++i) {
      dst[i] = TRANSPARENT_COLOR;
    }
  }
}

static void WebPUtilClearPic(WebPPicture* const picture,
                             const FrameRectangle* const rect) {
  if (rect != NULL) {
    ClearRectangle(picture, rect->x_offset, rect->y_offset,
                   rect->width, rect->height);
  } else {
    ClearRectangle(picture, 0, 0, picture->width, picture->height);
  }
}

static void MarkNoError(WebPAnimEncoder* const enc) {
  enc->error_str[0] = '\0';  // Empty string.
}

static void MarkError(WebPAnimEncoder* const enc, const char* str) {
  if (snprintf(enc->error_str, ERROR_STR_MAX_LENGTH, "%s.", str) < 0) {
    assert(0);  // FIX ME!
  }
}

static void MarkError2(WebPAnimEncoder* const enc,
                       const char* str, int error_code) {
  if (snprintf(enc->error_str, ERROR_STR_MAX_LENGTH, "%s: %d.", str,
               error_code) < 0) {
    assert(0);  // FIX ME!
  }
}

WebPAnimEncoder* WebPAnimEncoderNewInternal(
    int width, int height, const WebPAnimEncoderOptions* enc_options,
    int abi_version) {
  WebPAnimEncoder* enc;

  if (WEBP_ABI_IS_INCOMPATIBLE(abi_version, WEBP_MUX_ABI_VERSION)) {
    return NULL;
  }
  if (width <= 0 || height <= 0 ||
      (width * (uint64_t)height) >= MAX_IMAGE_AREA) {
    return NULL;
  }

  enc = (WebPAnimEncoder*)WebPSafeCalloc(1, sizeof(*enc));
  if (enc == NULL) return NULL;
  MarkNoError(enc);

  // Dimensions and options.
  *(int*)&enc->canvas_width = width;
  *(int*)&enc->canvas_height = height;
  if (enc_options != NULL) {
    *(WebPAnimEncoderOptions*)&enc->options = *enc_options;
    SanitizeEncoderOptions((WebPAnimEncoderOptions*)&enc->options);
  } else {
    DefaultEncoderOptions((WebPAnimEncoderOptions*)&enc->options);
  }

  // Canvas buffers.
  if (!WebPPictureInit(&enc->curr_canvas_copy) ||
      !WebPPictureInit(&enc->prev_canvas) ||
      !WebPPictureInit(&enc->prev_canvas_disposed)) {
    goto Err;
  }
  enc->curr_canvas_copy.width = width;
  enc->curr_canvas_copy.height = height;
  enc->curr_canvas_copy.use_argb = 1;
  if (!WebPPictureAlloc(&enc->curr_canvas_copy) ||
      !WebPPictureCopy(&enc->curr_canvas_copy, &enc->prev_canvas) ||
      !WebPPictureCopy(&enc->curr_canvas_copy, &enc->prev_canvas_disposed)) {
    goto Err;
  }
  WebPUtilClearPic(&enc->prev_canvas, NULL);
  enc->curr_canvas_copy_modified = 1;

  // Encoded frames.
  ResetCounters(enc);
  // Note: one extra storage is for the previous frame.
  enc->size = enc->options.kmax - enc->options.kmin + 1;
  // We need space for at least 2 frames. But when kmin, kmax are both zero,
  // enc->size will be 1. So we handle that special case below.
  if (enc->size < 2) enc->size = 2;
  enc->encoded_frames =
      (EncodedFrame*)WebPSafeCalloc(enc->size, sizeof(*enc->encoded_frames));
  if (enc->encoded_frames == NULL) goto Err;

  enc->mux = WebPMuxNew();
  if (enc->mux == NULL) goto Err;

  enc->count_since_key_frame = 0;
  enc->first_timestamp = 0;
  enc->prev_timestamp = 0;
  enc->prev_candidate_undecided = 0;
  enc->is_first_frame = 1;
  enc->got_null_frame = 0;

  return enc;  // All OK.

 Err:
  WebPAnimEncoderDelete(enc);
  return NULL;
}

// Release the data contained by 'encoded_frame'.
static void FrameRelease(EncodedFrame* const encoded_frame) {
  if (encoded_frame != NULL) {
    WebPDataClear(&encoded_frame->sub_frame.bitstream);
    WebPDataClear(&encoded_frame->key_frame.bitstream);
    memset(encoded_frame, 0, sizeof(*encoded_frame));
  }
}

void WebPAnimEncoderDelete(WebPAnimEncoder* enc) {
  if (enc != NULL) {
    WebPPictureFree(&enc->curr_canvas_copy);
    WebPPictureFree(&enc->prev_canvas);
    WebPPictureFree(&enc->prev_canvas_disposed);
    if (enc->encoded_frames != NULL) {
      size_t i;
      for (i = 0; i < enc->size; ++i) {
        FrameRelease(&enc->encoded_frames[i]);
      }
      WebPSafeFree(enc->encoded_frames);
    }
    WebPMuxDelete(enc->mux);
    WebPSafeFree(enc);
  }
}

// -----------------------------------------------------------------------------
// Frame addition.

// Returns cached frame at the given 'position'.
static EncodedFrame* GetFrame(const WebPAnimEncoder* const enc,
                              size_t position) {
  assert(enc->start + position < enc->size);
  return &enc->encoded_frames[enc->start + position];
}

typedef int (*ComparePixelsFunc)(const uint32_t*, int, const uint32_t*, int,
                                 int, int);

// Returns true if 'length' number of pixels in 'src' and 'dst' are equal,
// assuming the given step sizes between pixels.
// 'max_allowed_diff' is unused and only there to allow function pointer use.
static WEBP_INLINE int ComparePixelsLossless(const uint32_t* src, int src_step,
                                             const uint32_t* dst, int dst_step,
                                             int length, int max_allowed_diff) {
  (void)max_allowed_diff;
  assert(length > 0);
  while (length-- > 0) {
    if (*src != *dst) {
      return 0;
    }
    src += src_step;
    dst += dst_step;
  }
  return 1;
}

// Helper to check if each channel in 'src' and 'dst' is at most off by
// 'max_allowed_diff'.
static WEBP_INLINE int PixelsAreSimilar(uint32_t src, uint32_t dst,
                                        int max_allowed_diff) {
  const int src_a = (src >> 24) & 0xff;
  const int src_r = (src >> 16) & 0xff;
  const int src_g = (src >> 8) & 0xff;
  const int src_b = (src >> 0) & 0xff;
  const int dst_a = (dst >> 24) & 0xff;
  const int dst_r = (dst >> 16) & 0xff;
  const int dst_g = (dst >> 8) & 0xff;
  const int dst_b = (dst >> 0) & 0xff;

  return (src_a == dst_a) &&
         (abs(src_r - dst_r) * dst_a <= (max_allowed_diff * 255)) &&
         (abs(src_g - dst_g) * dst_a <= (max_allowed_diff * 255)) &&
         (abs(src_b - dst_b) * dst_a <= (max_allowed_diff * 255));
}

// Returns true if 'length' number of pixels in 'src' and 'dst' are within an
// error bound, assuming the given step sizes between pixels.
static WEBP_INLINE int ComparePixelsLossy(const uint32_t* src, int src_step,
                                          const uint32_t* dst, int dst_step,
                                          int length, int max_allowed_diff) {
  assert(length > 0);
  while (length-- > 0) {
    if (!PixelsAreSimilar(*src, *dst, max_allowed_diff)) {
      return 0;
    }
    src += src_step;
    dst += dst_step;
  }
  return 1;
}

static int IsEmptyRect(const FrameRectangle* const rect) {
  return (rect->width == 0) || (rect->height == 0);
}

static int QualityToMaxDiff(float quality) {
  const double val = pow(quality / 100., 0.5);
  const double max_diff = 31 * (1 - val) + 1 * val;
  return (int)(max_diff + 0.5);
}

// Assumes that an initial valid guess of change rectangle 'rect' is passed.
static void MinimizeChangeRectangle(const WebPPicture* const src,
                                    const WebPPicture* const dst,
                                    FrameRectangle* const rect,
                                    int is_lossless, float quality) {
  int i, j;
  const ComparePixelsFunc compare_pixels =
      is_lossless ? ComparePixelsLossless : ComparePixelsLossy;
  const int max_allowed_diff_lossy = QualityToMaxDiff(quality);
  const int max_allowed_diff = is_lossless ? 0 : max_allowed_diff_lossy;

  // Assumption/correctness checks.
  assert(src->width == dst->width && src->height == dst->height);
  assert(rect->x_offset + rect->width <= dst->width);
  assert(rect->y_offset + rect->height <= dst->height);

  // Left boundary.
  for (i = rect->x_offset; i < rect->x_offset + rect->width; ++i) {
    const uint32_t* const src_argb =
        &src->argb[rect->y_offset * src->argb_stride + i];
    const uint32_t* const dst_argb =
        &dst->argb[rect->y_offset * dst->argb_stride + i];
    if (compare_pixels(src_argb, src->argb_stride, dst_argb, dst->argb_stride,
                       rect->height, max_allowed_diff)) {
      --rect->width;  // Redundant column.
      ++rect->x_offset;
    } else {
      break;
    }
  }
  if (rect->width == 0) goto NoChange;

  // Right boundary.
  for (i = rect->x_offset + rect->width - 1; i >= rect->x_offset; --i) {
    const uint32_t* const src_argb =
        &src->argb[rect->y_offset * src->argb_stride + i];
    const uint32_t* const dst_argb =
        &dst->argb[rect->y_offset * dst->argb_stride + i];
    if (compare_pixels(src_argb, src->argb_stride, dst_argb, dst->argb_stride,
                       rect->height, max_allowed_diff)) {
      --rect->width;  // Redundant column.
    } else {
      break;
    }
  }
  if (rect->width == 0) goto NoChange;

  // Top boundary.
  for (j = rect->y_offset; j < rect->y_offset + rect->height; ++j) {
    const uint32_t* const src_argb =
        &src->argb[j * src->argb_stride + rect->x_offset];
    const uint32_t* const dst_argb =
        &dst->argb[j * dst->argb_stride + rect->x_offset];
    if (compare_pixels(src_argb, 1, dst_argb, 1, rect->width,
                       max_allowed_diff)) {
      --rect->height;  // Redundant row.
      ++rect->y_offset;
    } else {
      break;
    }
  }
  if (rect->height == 0) goto NoChange;

  // Bottom boundary.
  for (j = rect->y_offset + rect->height - 1; j >= rect->y_offset; --j) {
    const uint32_t* const src_argb =
        &src->argb[j * src->argb_stride + rect->x_offset];
    const uint32_t* const dst_argb =
        &dst->argb[j * dst->argb_stride + rect->x_offset];
    if (compare_pixels(src_argb, 1, dst_argb, 1, rect->width,
                       max_allowed_diff)) {
      --rect->height;  // Redundant row.
    } else {
      break;
    }
  }
  if (rect->height == 0) goto NoChange;

  if (IsEmptyRect(rect)) {
 NoChange:
    rect->x_offset = 0;
    rect->y_offset = 0;
    rect->width = 0;
    rect->height = 0;
  }
}

// Snap rectangle to even offsets (and adjust dimensions if needed).
static WEBP_INLINE void SnapToEvenOffsets(FrameRectangle* const rect) {
  rect->width += (rect->x_offset & 1);
  rect->height += (rect->y_offset & 1);
  rect->x_offset &= ~1;
  rect->y_offset &= ~1;
}

typedef struct {
  int should_try;                // Should try this set of parameters.
  int empty_rect_allowed;        // Frame with empty rectangle can be skipped.
  FrameRectangle rect_ll;        // Frame rectangle for lossless compression.
  WebPPicture sub_frame_ll;      // Sub-frame pic for lossless compression.
  FrameRectangle rect_lossy;     // Frame rectangle for lossy compression.
                                 // Could be smaller than 'rect_ll' as pixels
                                 // with small diffs can be ignored.
  WebPPicture sub_frame_lossy;   // Sub-frame pic for lossless compression.
} SubFrameParams;

static int SubFrameParamsInit(SubFrameParams* const params,
                              int should_try, int empty_rect_allowed) {
  params->should_try = should_try;
  params->empty_rect_allowed = empty_rect_allowed;
  if (!WebPPictureInit(&params->sub_frame_ll) ||
      !WebPPictureInit(&params->sub_frame_lossy)) {
    return 0;
  }
  return 1;
}

static void SubFrameParamsFree(SubFrameParams* const params) {
  WebPPictureFree(&params->sub_frame_ll);
  WebPPictureFree(&params->sub_frame_lossy);
}

// Given previous and current canvas, picks the optimal rectangle for the
// current frame based on 'is_lossless' and other parameters. Assumes that the
// initial guess 'rect' is valid.
static int GetSubRect(const WebPPicture* const prev_canvas,
                      const WebPPicture* const curr_canvas, int is_key_frame,
                      int is_first_frame, int empty_rect_allowed,
                      int is_lossless, float quality,
                      FrameRectangle* const rect,
                      WebPPicture* const sub_frame) {
  if (!is_key_frame || is_first_frame) {  // Optimize frame rectangle.
    // Note: This behaves as expected for first frame, as 'prev_canvas' is
    // initialized to a fully transparent canvas in the beginning.
    MinimizeChangeRectangle(prev_canvas, curr_canvas, rect,
                            is_lossless, quality);
  }

  if (IsEmptyRect(rect)) {
    if (empty_rect_allowed) {  // No need to get 'sub_frame'.
      return 1;
    } else {                   // Force a 1x1 rectangle.
      rect->width = 1;
      rect->height = 1;
      assert(rect->x_offset == 0);
      assert(rect->y_offset == 0);
    }
  }

  SnapToEvenOffsets(rect);
  return WebPPictureView(curr_canvas, rect->x_offset, rect->y_offset,
                         rect->width, rect->height, sub_frame);
}

// Picks optimal frame rectangle for both lossless and lossy compression. The
// initial guess for frame rectangles will be the full canvas.
static int GetSubRects(const WebPPicture* const prev_canvas,
                       const WebPPicture* const curr_canvas, int is_key_frame,
                       int is_first_frame, float quality,
                       SubFrameParams* const params) {
  // Lossless frame rectangle.
  params->rect_ll.x_offset = 0;
  params->rect_ll.y_offset = 0;
  params->rect_ll.width = curr_canvas->width;
  params->rect_ll.height = curr_canvas->height;
  if (!GetSubRect(prev_canvas, curr_canvas, is_key_frame, is_first_frame,
                  params->empty_rect_allowed, 1, quality,
                  &params->rect_ll, &params->sub_frame_ll)) {
    return 0;
  }
  // Lossy frame rectangle.
  params->rect_lossy = params->rect_ll;  // seed with lossless rect.
  return GetSubRect(prev_canvas, curr_canvas, is_key_frame, is_first_frame,
                    params->empty_rect_allowed, 0, quality,
                    &params->rect_lossy, &params->sub_frame_lossy);
}

static WEBP_INLINE int clip(int v, int min_v, int max_v) {
  return (v < min_v) ? min_v : (v > max_v) ? max_v : v;
}

int WebPAnimEncoderRefineRect(
    const WebPPicture* const prev_canvas, const WebPPicture* const curr_canvas,
    int is_lossless, float quality, int* const x_offset, int* const y_offset,
    int* const width, int* const height) {
  FrameRectangle rect;
  int right, left, bottom, top;
  if (prev_canvas == NULL || curr_canvas == NULL ||
      prev_canvas->width != curr_canvas->width ||
      prev_canvas->height != curr_canvas->height ||
      !prev_canvas->use_argb || !curr_canvas->use_argb) {
    return 0;
  }
  right = clip(*x_offset + *width, 0, curr_canvas->width);
  left = clip(*x_offset, 0, curr_canvas->width - 1);
  bottom = clip(*y_offset + *height, 0, curr_canvas->height);
  top = clip(*y_offset, 0, curr_canvas->height - 1);
  rect.x_offset = left;
  rect.y_offset = top;
  rect.width = clip(right - left, 0, curr_canvas->width - rect.x_offset);
  rect.height = clip(bottom - top, 0, curr_canvas->height - rect.y_offset);
  MinimizeChangeRectangle(prev_canvas, curr_canvas, &rect, is_lossless,
                          quality);
  SnapToEvenOffsets(&rect);
  *x_offset = rect.x_offset;
  *y_offset = rect.y_offset;
  *width = rect.width;
  *height = rect.height;
  return 1;
}

static void DisposeFrameRectangle(int dispose_method,
                                  const FrameRectangle* const rect,
                                  WebPPicture* const curr_canvas) {
  assert(rect != NULL);
  if (dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
    WebPUtilClearPic(curr_canvas, rect);
  }
}

static uint32_t RectArea(const FrameRectangle* const rect) {
  return (uint32_t)rect->width * rect->height;
}

static int IsLosslessBlendingPossible(const WebPPicture* const src,
                                      const WebPPicture* const dst,
                                      const FrameRectangle* const rect) {
  int i, j;
  assert(src->width == dst->width && src->height == dst->height);
  assert(rect->x_offset + rect->width <= dst->width);
  assert(rect->y_offset + rect->height <= dst->height);
  for (j = rect->y_offset; j < rect->y_offset + rect->height; ++j) {
    for (i = rect->x_offset; i < rect->x_offset + rect->width; ++i) {
      const uint32_t src_pixel = src->argb[j * src->argb_stride + i];
      const uint32_t dst_pixel = dst->argb[j * dst->argb_stride + i];
      const uint32_t dst_alpha = dst_pixel >> 24;
      if (dst_alpha != 0xff && src_pixel != dst_pixel) {
        // In this case, if we use blending, we can't attain the desired
        // 'dst_pixel' value for this pixel. So, blending is not possible.
        return 0;
      }
    }
  }
  return 1;
}

static int IsLossyBlendingPossible(const WebPPicture* const src,
                                   const WebPPicture* const dst,
                                   const FrameRectangle* const rect,
                                   float quality) {
  const int max_allowed_diff_lossy = QualityToMaxDiff(quality);
  int i, j;
  assert(src->width == dst->width && src->height == dst->height);
  assert(rect->x_offset + rect->width <= dst->width);
  assert(rect->y_offset + rect->height <= dst->height);
  for (j = rect->y_offset; j < rect->y_offset + rect->height; ++j) {
    for (i = rect->x_offset; i < rect->x_offset + rect->width; ++i) {
      const uint32_t src_pixel = src->argb[j * src->argb_stride + i];
      const uint32_t dst_pixel = dst->argb[j * dst->argb_stride + i];
      const uint32_t dst_alpha = dst_pixel >> 24;
      if (dst_alpha != 0xff &&
          !PixelsAreSimilar(src_pixel, dst_pixel, max_allowed_diff_lossy)) {
        // In this case, if we use blending, we can't attain the desired
        // 'dst_pixel' value for this pixel. So, blending is not possible.
        return 0;
      }
    }
  }
  return 1;
}

// For pixels in 'rect', replace those pixels in 'dst' that are same as 'src' by
// transparent pixels.
// Returns true if at least one pixel gets modified.
static int IncreaseTransparency(const WebPPicture* const src,
                                const FrameRectangle* const rect,
                                WebPPicture* const dst) {
  int i, j;
  int modified = 0;
  assert(src != NULL && dst != NULL && rect != NULL);
  assert(src->width == dst->width && src->height == dst->height);
  for (j = rect->y_offset; j < rect->y_offset + rect->height; ++j) {
    const uint32_t* const psrc = src->argb + j * src->argb_stride;
    uint32_t* const pdst = dst->argb + j * dst->argb_stride;
    for (i = rect->x_offset; i < rect->x_offset + rect->width; ++i) {
      if (psrc[i] == pdst[i] && pdst[i] != TRANSPARENT_COLOR) {
        pdst[i] = TRANSPARENT_COLOR;
        modified = 1;
      }
    }
  }
  return modified;
}

#undef TRANSPARENT_COLOR

// Replace similar blocks of pixels by a 'see-through' transparent block
// with uniform average color.
// Assumes lossy compression is being used.
// Returns true if at least one pixel gets modified.
static int FlattenSimilarBlocks(const WebPPicture* const src,
                                const FrameRectangle* const rect,
                                WebPPicture* const dst, float quality) {
  const int max_allowed_diff_lossy = QualityToMaxDiff(quality);
  int i, j;
  int modified = 0;
  const int block_size = 8;
  const int y_start = (rect->y_offset + block_size) & ~(block_size - 1);
  const int y_end = (rect->y_offset + rect->height) & ~(block_size - 1);
  const int x_start = (rect->x_offset + block_size) & ~(block_size - 1);
  const int x_end = (rect->x_offset + rect->width) & ~(block_size - 1);
  assert(src != NULL && dst != NULL && rect != NULL);
  assert(src->width == dst->width && src->height == dst->height);
  assert((block_size & (block_size - 1)) == 0);  // must be a power of 2
  // Iterate over each block and count similar pixels.
  for (j = y_start; j < y_end; j += block_size) {
    for (i = x_start; i < x_end; i += block_size) {
      int cnt = 0;
      int avg_r = 0, avg_g = 0, avg_b = 0;
      int x, y;
      const uint32_t* const psrc = src->argb + j * src->argb_stride + i;
      uint32_t* const pdst = dst->argb + j * dst->argb_stride + i;
      for (y = 0; y < block_size; ++y) {
        for (x = 0; x < block_size; ++x) {
          const uint32_t src_pixel = psrc[x + y * src->argb_stride];
          const int alpha = src_pixel >> 24;
          if (alpha == 0xff &&
              PixelsAreSimilar(src_pixel, pdst[x + y * dst->argb_stride],
                               max_allowed_diff_lossy)) {
            ++cnt;
            avg_r += (src_pixel >> 16) & 0xff;
            avg_g += (src_pixel >> 8) & 0xff;
            avg_b += (src_pixel >> 0) & 0xff;
          }
        }
      }
      // If we have a fully similar block, we replace it with an
      // average transparent block. This compresses better in lossy mode.
      if (cnt == block_size * block_size) {
        const uint32_t color = (0x00          << 24) |
                               ((avg_r / cnt) << 16) |
                               ((avg_g / cnt) <<  8) |
                               ((avg_b / cnt) <<  0);
        for (y = 0; y < block_size; ++y) {
          for (x = 0; x < block_size; ++x) {
            pdst[x + y * dst->argb_stride] = color;
          }
        }
        modified = 1;
      }
    }
  }
  return modified;
}

static int EncodeFrame(const WebPConfig* const config, WebPPicture* const pic,
                       WebPMemoryWriter* const memory) {
  pic->use_argb = 1;
  pic->writer = WebPMemoryWrite;
  pic->custom_ptr = memory;
  if (!WebPEncode(config, pic)) {
    return 0;
  }
  return 1;
}

// Struct representing a candidate encoded frame including its metadata.
typedef struct {
  WebPMemoryWriter  mem;
  WebPMuxFrameInfo  info;
  FrameRectangle    rect;
  int               evaluate;  // True if this candidate should be evaluated.
} Candidate;

// Generates a candidate encoded frame given a picture and metadata.
static WebPEncodingError EncodeCandidate(WebPPicture* const sub_frame,
                                         const FrameRectangle* const rect,
                                         const WebPConfig* const encoder_config,
                                         int use_blending,
                                         Candidate* const candidate) {
  WebPConfig config = *encoder_config;
  WebPEncodingError error_code = VP8_ENC_OK;
  assert(candidate != NULL);
  memset(candidate, 0, sizeof(*candidate));

  // Set frame rect and info.
  candidate->rect = *rect;
  candidate->info.id = WEBP_CHUNK_ANMF;
  candidate->info.x_offset = rect->x_offset;
  candidate->info.y_offset = rect->y_offset;
  candidate->info.dispose_method = WEBP_MUX_DISPOSE_NONE;  // Set later.
  candidate->info.blend_method =
      use_blending ? WEBP_MUX_BLEND : WEBP_MUX_NO_BLEND;
  candidate->info.duration = 0;  // Set in next call to WebPAnimEncoderAdd().

  // Encode picture.
  WebPMemoryWriterInit(&candidate->mem);

  if (!config.lossless && use_blending) {
    // Disable filtering to avoid blockiness in reconstructed frames at the
    // time of decoding.
    config.autofilter = 0;
    config.filter_strength = 0;
  }
  if (!EncodeFrame(&config, sub_frame, &candidate->mem)) {
    error_code = sub_frame->error_code;
    goto Err;
  }

  candidate->evaluate = 1;
  return error_code;

 Err:
  WebPMemoryWriterClear(&candidate->mem);
  return error_code;
}

static void CopyCurrentCanvas(WebPAnimEncoder* const enc) {
  if (enc->curr_canvas_copy_modified) {
    WebPCopyPixels(enc->curr_canvas, &enc->curr_canvas_copy);
    enc->curr_canvas_copy.progress_hook = enc->curr_canvas->progress_hook;
    enc->curr_canvas_copy.user_data = enc->curr_canvas->user_data;
    enc->curr_canvas_copy_modified = 0;
  }
}

enum {
  LL_DISP_NONE = 0,
  LL_DISP_BG,
  LOSSY_DISP_NONE,
  LOSSY_DISP_BG,
  CANDIDATE_COUNT
};

#define MIN_COLORS_LOSSY     31  // Don't try lossy below this threshold.
#define MAX_COLORS_LOSSLESS 194  // Don't try lossless above this threshold.

// Generates candidates for a given dispose method given pre-filled sub-frame
// 'params'.
static WebPEncodingError GenerateCandidates(
    WebPAnimEncoder* const enc, Candidate candidates[CANDIDATE_COUNT],
    WebPMuxAnimDispose dispose_method, int is_lossless, int is_key_frame,
    SubFrameParams* const params,
    const WebPConfig* const config_ll, const WebPConfig* const config_lossy) {
  WebPEncodingError error_code = VP8_ENC_OK;
  const int is_dispose_none = (dispose_method == WEBP_MUX_DISPOSE_NONE);
  Candidate* const candidate_ll =
      is_dispose_none ? &candidates[LL_DISP_NONE] : &candidates[LL_DISP_BG];
  Candidate* const candidate_lossy = is_dispose_none
                                     ? &candidates[LOSSY_DISP_NONE]
                                     : &candidates[LOSSY_DISP_BG];
  WebPPicture* const curr_canvas = &enc->curr_canvas_copy;
  const WebPPicture* const prev_canvas =
      is_dispose_none ? &enc->prev_canvas : &enc->prev_canvas_disposed;
  int use_blending_ll, use_blending_lossy;
  int evaluate_ll, evaluate_lossy;

  CopyCurrentCanvas(enc);
  use_blending_ll =
      !is_key_frame &&
      IsLosslessBlendingPossible(prev_canvas, curr_canvas, &params->rect_ll);
  use_blending_lossy =
      !is_key_frame &&
      IsLossyBlendingPossible(prev_canvas, curr_canvas, &params->rect_lossy,
                              config_lossy->quality);

  // Pick candidates to be tried.
  if (!enc->options.allow_mixed) {
    evaluate_ll = is_lossless;
    evaluate_lossy = !is_lossless;
  } else if (enc->options.minimize_size) {
    evaluate_ll = 1;
    evaluate_lossy = 1;
  } else {  // Use a heuristic for trying lossless and/or lossy compression.
    const int num_colors = WebPGetColorPalette(&params->sub_frame_ll, NULL);
    evaluate_ll = (num_colors < MAX_COLORS_LOSSLESS);
    evaluate_lossy = (num_colors >= MIN_COLORS_LOSSY);
  }

  // Generate candidates.
  if (evaluate_ll) {
    CopyCurrentCanvas(enc);
    if (use_blending_ll) {
      enc->curr_canvas_copy_modified =
          IncreaseTransparency(prev_canvas, &params->rect_ll, curr_canvas);
    }
    error_code = EncodeCandidate(&params->sub_frame_ll, &params->rect_ll,
                                 config_ll, use_blending_ll, candidate_ll);
    if (error_code != VP8_ENC_OK) return error_code;
  }
  if (evaluate_lossy) {
    CopyCurrentCanvas(enc);
    if (use_blending_lossy) {
      enc->curr_canvas_copy_modified =
          FlattenSimilarBlocks(prev_canvas, &params->rect_lossy, curr_canvas,
                               config_lossy->quality);
    }
    error_code =
        EncodeCandidate(&params->sub_frame_lossy, &params->rect_lossy,
                        config_lossy, use_blending_lossy, candidate_lossy);
    if (error_code != VP8_ENC_OK) return error_code;
    enc->curr_canvas_copy_modified = 1;
  }
  return error_code;
}

#undef MIN_COLORS_LOSSY
#undef MAX_COLORS_LOSSLESS

static void GetEncodedData(const WebPMemoryWriter* const memory,
                           WebPData* const encoded_data) {
  encoded_data->bytes = memory->mem;
  encoded_data->size  = memory->size;
}

// Sets dispose method of the previous frame to be 'dispose_method'.
static void SetPreviousDisposeMethod(WebPAnimEncoder* const enc,
                                     WebPMuxAnimDispose dispose_method) {
  const size_t position = enc->count - 2;
  EncodedFrame* const prev_enc_frame = GetFrame(enc, position);
  assert(enc->count >= 2);  // As current and previous frames are in enc.

  if (enc->prev_candidate_undecided) {
    assert(dispose_method == WEBP_MUX_DISPOSE_NONE);
    prev_enc_frame->sub_frame.dispose_method = dispose_method;
    prev_enc_frame->key_frame.dispose_method = dispose_method;
  } else {
    WebPMuxFrameInfo* const prev_info = prev_enc_frame->is_key_frame
                                        ? &prev_enc_frame->key_frame
                                        : &prev_enc_frame->sub_frame;
    prev_info->dispose_method = dispose_method;
  }
}

static int IncreasePreviousDuration(WebPAnimEncoder* const enc, int duration) {
  const size_t position = enc->count - 1;
  EncodedFrame* const prev_enc_frame = GetFrame(enc, position);
  int new_duration;

  assert(enc->count >= 1);
  assert(!prev_enc_frame->is_key_frame ||
         prev_enc_frame->sub_frame.duration ==
         prev_enc_frame->key_frame.duration);
  assert(prev_enc_frame->sub_frame.duration ==
         (prev_enc_frame->sub_frame.duration & (MAX_DURATION - 1)));
  assert(duration == (duration & (MAX_DURATION - 1)));

  new_duration = prev_enc_frame->sub_frame.duration + duration;
  if (new_duration >= MAX_DURATION) {  // Special case.
    // Separate out previous frame from earlier merged frames to avoid overflow.
    // We add a 1x1 transparent frame for the previous frame, with blending on.
    const FrameRectangle rect = { 0, 0, 1, 1 };
    const uint8_t lossless_1x1_bytes[] = {
      0x52, 0x49, 0x46, 0x46, 0x14, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x4c, 0x08, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
      0x10, 0x88, 0x88, 0x08
    };
    const WebPData lossless_1x1 = {
      lossless_1x1_bytes, sizeof(lossless_1x1_bytes)
    };
    const uint8_t lossy_1x1_bytes[] = {
      0x52, 0x49, 0x46, 0x46, 0x40, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x58, 0x0a, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x4c, 0x50, 0x48, 0x02, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x20, 0x18, 0x00, 0x00, 0x00,
      0x30, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x01, 0x00, 0x01, 0x00, 0x02, 0x00,
      0x34, 0x25, 0xa4, 0x00, 0x03, 0x70, 0x00, 0xfe, 0xfb, 0xfd, 0x50, 0x00
    };
    const WebPData lossy_1x1 = { lossy_1x1_bytes, sizeof(lossy_1x1_bytes) };
    const int can_use_lossless =
        (enc->last_config.lossless || enc->options.allow_mixed);
    EncodedFrame* const curr_enc_frame = GetFrame(enc, enc->count);
    curr_enc_frame->is_key_frame = 0;
    curr_enc_frame->sub_frame.id = WEBP_CHUNK_ANMF;
    curr_enc_frame->sub_frame.x_offset = 0;
    curr_enc_frame->sub_frame.y_offset = 0;
    curr_enc_frame->sub_frame.dispose_method = WEBP_MUX_DISPOSE_NONE;
    curr_enc_frame->sub_frame.blend_method = WEBP_MUX_BLEND;
    curr_enc_frame->sub_frame.duration = duration;
    if (!WebPDataCopy(can_use_lossless ? &lossless_1x1 : &lossy_1x1,
                      &curr_enc_frame->sub_frame.bitstream)) {
      return 0;
    }
    ++enc->count;
    ++enc->count_since_key_frame;
    enc->flush_count = enc->count - 1;
    enc->prev_candidate_undecided = 0;
    enc->prev_rect = rect;
  } else {                           // Regular case.
    // Increase duration of the previous frame by 'duration'.
    prev_enc_frame->sub_frame.duration = new_duration;
    prev_enc_frame->key_frame.duration = new_duration;
  }
  return 1;
}

// Pick the candidate encoded frame with smallest size and release other
// candidates.
// TODO(later): Perhaps a rough SSIM/PSNR produced by the encoder should
// also be a criteria, in addition to sizes.
static void PickBestCandidate(WebPAnimEncoder* const enc,
                              Candidate* const candidates, int is_key_frame,
                              EncodedFrame* const encoded_frame) {
  int i;
  int best_idx = -1;
  size_t best_size = ~0;
  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    if (candidates[i].evaluate) {
      const size_t candidate_size = candidates[i].mem.size;
      if (candidate_size < best_size) {
        best_idx = i;
        best_size = candidate_size;
      }
    }
  }
  assert(best_idx != -1);
  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    if (candidates[i].evaluate) {
      if (i == best_idx) {
        WebPMuxFrameInfo* const dst = is_key_frame
                                      ? &encoded_frame->key_frame
                                      : &encoded_frame->sub_frame;
        *dst = candidates[i].info;
        GetEncodedData(&candidates[i].mem, &dst->bitstream);
        if (!is_key_frame) {
          // Note: Previous dispose method only matters for non-keyframes.
          // Also, we don't want to modify previous dispose method that was
          // selected when a non key-frame was assumed.
          const WebPMuxAnimDispose prev_dispose_method =
              (best_idx == LL_DISP_NONE || best_idx == LOSSY_DISP_NONE)
                  ? WEBP_MUX_DISPOSE_NONE
                  : WEBP_MUX_DISPOSE_BACKGROUND;
          SetPreviousDisposeMethod(enc, prev_dispose_method);
        }
        enc->prev_rect = candidates[i].rect;  // save for next frame.
      } else {
        WebPMemoryWriterClear(&candidates[i].mem);
        candidates[i].evaluate = 0;
      }
    }
  }
}

// Depending on the configuration, tries different compressions
// (lossy/lossless), dispose methods, blending methods etc to encode the current
// frame and outputs the best one in 'encoded_frame'.
// 'frame_skipped' will be set to true if this frame should actually be skipped.
static WebPEncodingError SetFrame(WebPAnimEncoder* const enc,
                                  const WebPConfig* const config,
                                  int is_key_frame,
                                  EncodedFrame* const encoded_frame,
                                  int* const frame_skipped) {
  int i;
  WebPEncodingError error_code = VP8_ENC_OK;
  const WebPPicture* const curr_canvas = &enc->curr_canvas_copy;
  const WebPPicture* const prev_canvas = &enc->prev_canvas;
  Candidate candidates[CANDIDATE_COUNT];
  const int is_lossless = config->lossless;
  const int consider_lossless = is_lossless || enc->options.allow_mixed;
  const int consider_lossy = !is_lossless || enc->options.allow_mixed;
  const int is_first_frame = enc->is_first_frame;

  // First frame cannot be skipped as there is no 'previous frame' to merge it
  // to. So, empty rectangle is not allowed for the first frame.
  const int empty_rect_allowed_none = !is_first_frame;

  // Even if there is exact pixel match between 'disposed previous canvas' and
  // 'current canvas', we can't skip current frame, as there may not be exact
  // pixel match between 'previous canvas' and 'current canvas'. So, we don't
  // allow empty rectangle in this case.
  const int empty_rect_allowed_bg = 0;

  // If current frame is a key-frame, dispose method of previous frame doesn't
  // matter, so we don't try dispose to background.
  // Also, if key-frame insertion is on, and previous frame could be picked as
  // either a sub-frame or a key-frame, then we can't be sure about what frame
  // rectangle would be disposed. In that case too, we don't try dispose to
  // background.
  const int dispose_bg_possible =
      !is_key_frame && !enc->prev_candidate_undecided;

  SubFrameParams dispose_none_params;
  SubFrameParams dispose_bg_params;

  WebPConfig config_ll = *config;
  WebPConfig config_lossy = *config;
  config_ll.lossless = 1;
  config_lossy.lossless = 0;
  enc->last_config = *config;
  enc->last_config_reversed = config->lossless ? config_lossy : config_ll;
  *frame_skipped = 0;

  if (!SubFrameParamsInit(&dispose_none_params, 1, empty_rect_allowed_none) ||
      !SubFrameParamsInit(&dispose_bg_params, 0, empty_rect_allowed_bg)) {
    return VP8_ENC_ERROR_INVALID_CONFIGURATION;
  }

  memset(candidates, 0, sizeof(candidates));

  // Change-rectangle assuming previous frame was DISPOSE_NONE.
  if (!GetSubRects(prev_canvas, curr_canvas, is_key_frame, is_first_frame,
                   config_lossy.quality, &dispose_none_params)) {
    error_code = VP8_ENC_ERROR_INVALID_CONFIGURATION;
    goto Err;
  }

  if ((consider_lossless && IsEmptyRect(&dispose_none_params.rect_ll)) ||
      (consider_lossy && IsEmptyRect(&dispose_none_params.rect_lossy))) {
    // Don't encode the frame at all. Instead, the duration of the previous
    // frame will be increased later.
    assert(empty_rect_allowed_none);
    *frame_skipped = 1;
    goto End;
  }

  if (dispose_bg_possible) {
    // Change-rectangle assuming previous frame was DISPOSE_BACKGROUND.
    WebPPicture* const prev_canvas_disposed = &enc->prev_canvas_disposed;
    WebPCopyPixels(prev_canvas, prev_canvas_disposed);
    DisposeFrameRectangle(WEBP_MUX_DISPOSE_BACKGROUND, &enc->prev_rect,
                          prev_canvas_disposed);

    if (!GetSubRects(prev_canvas_disposed, curr_canvas, is_key_frame,
                     is_first_frame, config_lossy.quality,
                     &dispose_bg_params)) {
      error_code = VP8_ENC_ERROR_INVALID_CONFIGURATION;
      goto Err;
    }
    assert(!IsEmptyRect(&dispose_bg_params.rect_ll));
    assert(!IsEmptyRect(&dispose_bg_params.rect_lossy));

    if (enc->options.minimize_size) {  // Try both dispose methods.
      dispose_bg_params.should_try = 1;
      dispose_none_params.should_try = 1;
    } else if ((is_lossless &&
                RectArea(&dispose_bg_params.rect_ll) <
                    RectArea(&dispose_none_params.rect_ll)) ||
               (!is_lossless &&
                RectArea(&dispose_bg_params.rect_lossy) <
                    RectArea(&dispose_none_params.rect_lossy))) {
      dispose_bg_params.should_try = 1;  // Pick DISPOSE_BACKGROUND.
      dispose_none_params.should_try = 0;
    }
  }

  if (dispose_none_params.should_try) {
    error_code = GenerateCandidates(
        enc, candidates, WEBP_MUX_DISPOSE_NONE, is_lossless, is_key_frame,
        &dispose_none_params, &config_ll, &config_lossy);
    if (error_code != VP8_ENC_OK) goto Err;
  }

  if (dispose_bg_params.should_try) {
    assert(!enc->is_first_frame);
    assert(dispose_bg_possible);
    error_code = GenerateCandidates(
        enc, candidates, WEBP_MUX_DISPOSE_BACKGROUND, is_lossless, is_key_frame,
        &dispose_bg_params, &config_ll, &config_lossy);
    if (error_code != VP8_ENC_OK) goto Err;
  }

  PickBestCandidate(enc, candidates, is_key_frame, encoded_frame);

  goto End;

 Err:
  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    if (candidates[i].evaluate) {
      WebPMemoryWriterClear(&candidates[i].mem);
    }
  }

 End:
  SubFrameParamsFree(&dispose_none_params);
  SubFrameParamsFree(&dispose_bg_params);
  return error_code;
}

// Calculate the penalty incurred if we encode given frame as a key frame
// instead of a sub-frame.
static int64_t KeyFramePenalty(const EncodedFrame* const encoded_frame) {
  return ((int64_t)encoded_frame->key_frame.bitstream.size -
          encoded_frame->sub_frame.bitstream.size);
}

static int CacheFrame(WebPAnimEncoder* const enc,
                      const WebPConfig* const config) {
  int ok = 0;
  int frame_skipped = 0;
  WebPEncodingError error_code = VP8_ENC_OK;
  const size_t position = enc->count;
  EncodedFrame* const encoded_frame = GetFrame(enc, position);

  ++enc->count;

  if (enc->is_first_frame) {  // Add this as a key-frame.
    error_code = SetFrame(enc, config, 1, encoded_frame, &frame_skipped);
    if (error_code != VP8_ENC_OK) goto End;
    assert(frame_skipped == 0);  // First frame can't be skipped, even if empty.
    assert(position == 0 && enc->count == 1);
    encoded_frame->is_key_frame = 1;
    enc->flush_count = 0;
    enc->count_since_key_frame = 0;
    enc->prev_candidate_undecided = 0;
  } else {
    ++enc->count_since_key_frame;
    if (enc->count_since_key_frame <= enc->options.kmin) {
      // Add this as a frame rectangle.
      error_code = SetFrame(enc, config, 0, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      if (frame_skipped) goto Skip;
      encoded_frame->is_key_frame = 0;
      enc->flush_count = enc->count - 1;
      enc->prev_candidate_undecided = 0;
    } else {
      int64_t curr_delta;
      FrameRectangle prev_rect_key, prev_rect_sub;

      // Add this as a frame rectangle to enc.
      error_code = SetFrame(enc, config, 0, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      if (frame_skipped) goto Skip;
      prev_rect_sub = enc->prev_rect;


      // Add this as a key-frame to enc, too.
      error_code = SetFrame(enc, config, 1, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      assert(frame_skipped == 0);  // Key-frame cannot be an empty rectangle.
      prev_rect_key = enc->prev_rect;

      // Analyze size difference of the two variants.
      curr_delta = KeyFramePenalty(encoded_frame);
      if (curr_delta <= enc->best_delta) {  // Pick this as the key-frame.
        if (enc->keyframe != KEYFRAME_NONE) {
          EncodedFrame* const old_keyframe = GetFrame(enc, enc->keyframe);
          assert(old_keyframe->is_key_frame);
          old_keyframe->is_key_frame = 0;
        }
        encoded_frame->is_key_frame = 1;
        enc->prev_candidate_undecided = 1;
        enc->keyframe = (int)position;
        enc->best_delta = curr_delta;
        enc->flush_count = enc->count - 1;  // We can flush previous frames.
      } else {
        encoded_frame->is_key_frame = 0;
        enc->prev_candidate_undecided = 0;
      }
      // Note: We need '>=' below because when kmin and kmax are both zero,
      // count_since_key_frame will always be > kmax.
      if (enc->count_since_key_frame >= enc->options.kmax) {
        enc->flush_count = enc->count - 1;
        enc->count_since_key_frame = 0;
        enc->keyframe = KEYFRAME_NONE;
        enc->best_delta = DELTA_INFINITY;
      }
      if (!enc->prev_candidate_undecided) {
        enc->prev_rect =
            encoded_frame->is_key_frame ? prev_rect_key : prev_rect_sub;
      }
    }
  }

  // Update previous to previous and previous canvases for next call.
  WebPCopyPixels(enc->curr_canvas, &enc->prev_canvas);
  enc->is_first_frame = 0;

 Skip:
  ok = 1;
  ++enc->in_frame_count;

 End:
  if (!ok || frame_skipped) {
    FrameRelease(encoded_frame);
    // We reset some counters, as the frame addition failed/was skipped.
    --enc->count;
    if (!enc->is_first_frame) --enc->count_since_key_frame;
    if (!ok) {
      MarkError2(enc, "ERROR adding frame. WebPEncodingError", error_code);
    }
  }
  enc->curr_canvas->error_code = error_code;   // report error_code
  assert(ok || error_code != VP8_ENC_OK);
  return ok;
}

static int FlushFrames(WebPAnimEncoder* const enc) {
  while (enc->flush_count > 0) {
    WebPMuxError err;
    EncodedFrame* const curr = GetFrame(enc, 0);
    const WebPMuxFrameInfo* const info =
        curr->is_key_frame ? &curr->key_frame : &curr->sub_frame;
    assert(enc->mux != NULL);
    err = WebPMuxPushFrame(enc->mux, info, 1);
    if (err != WEBP_MUX_OK) {
      MarkError2(enc, "ERROR adding frame. WebPMuxError", err);
      return 0;
    }
    if (enc->options.verbose) {
      fprintf(stderr, "INFO: Added frame. offset:%d,%d dispose:%d blend:%d\n",
              info->x_offset, info->y_offset, info->dispose_method,
              info->blend_method);
    }
    ++enc->out_frame_count;
    FrameRelease(curr);
    ++enc->start;
    --enc->flush_count;
    --enc->count;
    if (enc->keyframe != KEYFRAME_NONE) --enc->keyframe;
  }

  if (enc->count == 1 && enc->start != 0) {
    // Move enc->start to index 0.
    const int enc_start_tmp = (int)enc->start;
    EncodedFrame temp = enc->encoded_frames[0];
    enc->encoded_frames[0] = enc->encoded_frames[enc_start_tmp];
    enc->encoded_frames[enc_start_tmp] = temp;
    FrameRelease(&enc->encoded_frames[enc_start_tmp]);
    enc->start = 0;
  }
  return 1;
}

#undef DELTA_INFINITY
#undef KEYFRAME_NONE

int WebPAnimEncoderAdd(WebPAnimEncoder* enc, WebPPicture* frame, int timestamp,
                       const WebPConfig* encoder_config) {
  WebPConfig config;
  int ok;

  if (enc == NULL) {
    return 0;
  }
  MarkNoError(enc);

  if (!enc->is_first_frame) {
    // Make sure timestamps are non-decreasing (integer wrap-around is OK).
    const uint32_t prev_frame_duration =
        (uint32_t)timestamp - enc->prev_timestamp;
    if (prev_frame_duration >= MAX_DURATION) {
      if (frame != NULL) {
        frame->error_code = VP8_ENC_ERROR_INVALID_CONFIGURATION;
      }
      MarkError(enc, "ERROR adding frame: timestamps must be non-decreasing");
      return 0;
    }
    if (!IncreasePreviousDuration(enc, (int)prev_frame_duration)) {
      return 0;
    }
    // IncreasePreviousDuration() may add a frame to avoid exceeding
    // MAX_DURATION which could cause CacheFrame() to over read 'encoded_frames'
    // before the next flush.
    if (enc->count == enc->size && !FlushFrames(enc)) {
      return 0;
    }
  } else {
    enc->first_timestamp = timestamp;
  }

  if (frame == NULL) {  // Special: last call.
    enc->got_null_frame = 1;
    enc->prev_timestamp = timestamp;
    return 1;
  }

  if (frame->width != enc->canvas_width ||
      frame->height != enc->canvas_height) {
    frame->error_code = VP8_ENC_ERROR_INVALID_CONFIGURATION;
    MarkError(enc, "ERROR adding frame: Invalid frame dimensions");
    return 0;
  }

  if (!frame->use_argb) {  // Convert frame from YUV(A) to ARGB.
    if (enc->options.verbose) {
      fprintf(stderr, "WARNING: Converting frame from YUV(A) to ARGB format; "
              "this incurs a small loss.\n");
    }
    if (!WebPPictureYUVAToARGB(frame)) {
      MarkError(enc, "ERROR converting frame from YUV(A) to ARGB");
      return 0;
    }
  }

  if (encoder_config != NULL) {
    if (!WebPValidateConfig(encoder_config)) {
      MarkError(enc, "ERROR adding frame: Invalid WebPConfig");
      return 0;
    }
    config = *encoder_config;
  } else {
    if (!WebPConfigInit(&config)) {
      MarkError(enc, "Cannot Init config");
      return 0;
    }
    config.lossless = 1;
  }
  assert(enc->curr_canvas == NULL);
  enc->curr_canvas = frame;  // Store reference.
  assert(enc->curr_canvas_copy_modified == 1);
  CopyCurrentCanvas(enc);

  ok = CacheFrame(enc, &config) && FlushFrames(enc);

  enc->curr_canvas = NULL;
  enc->curr_canvas_copy_modified = 1;
  if (ok) {
    enc->prev_timestamp = timestamp;
  }
  return ok;
}

// -----------------------------------------------------------------------------
// Bitstream assembly.

WEBP_NODISCARD static int DecodeFrameOntoCanvas(
    const WebPMuxFrameInfo* const frame, WebPPicture* const canvas) {
  const WebPData* const image = &frame->bitstream;
  WebPPicture sub_image;
  WebPDecoderConfig config;
  if (!WebPInitDecoderConfig(&config)) {
    return 0;
  }
  WebPUtilClearPic(canvas, NULL);
  if (WebPGetFeatures(image->bytes, image->size, &config.input) !=
      VP8_STATUS_OK) {
    return 0;
  }
  if (!WebPPictureView(canvas, frame->x_offset, frame->y_offset,
                       config.input.width, config.input.height, &sub_image)) {
    return 0;
  }
  config.output.is_external_memory = 1;
  config.output.colorspace = MODE_BGRA;
  config.output.u.RGBA.rgba = (uint8_t*)sub_image.argb;
  config.output.u.RGBA.stride = sub_image.argb_stride * 4;
  config.output.u.RGBA.size = config.output.u.RGBA.stride * sub_image.height;

  if (WebPDecode(image->bytes, image->size, &config) != VP8_STATUS_OK) {
    return 0;
  }
  return 1;
}

static int FrameToFullCanvas(WebPAnimEncoder* const enc,
                             const WebPMuxFrameInfo* const frame,
                             WebPData* const full_image) {
  WebPPicture* const canvas_buf = &enc->curr_canvas_copy;
  WebPMemoryWriter mem1, mem2;
  WebPMemoryWriterInit(&mem1);
  WebPMemoryWriterInit(&mem2);

  if (!DecodeFrameOntoCanvas(frame, canvas_buf)) goto Err;
  if (!EncodeFrame(&enc->last_config, canvas_buf, &mem1)) goto Err;
  GetEncodedData(&mem1, full_image);

  if (enc->options.allow_mixed) {
    if (!EncodeFrame(&enc->last_config_reversed, canvas_buf, &mem2)) goto Err;
    if (mem2.size < mem1.size) {
      GetEncodedData(&mem2, full_image);
      WebPMemoryWriterClear(&mem1);
    } else {
      WebPMemoryWriterClear(&mem2);
    }
  }
  return 1;

 Err:
  WebPMemoryWriterClear(&mem1);
  WebPMemoryWriterClear(&mem2);
  return 0;
}

// Convert a single-frame animation to a non-animated image if appropriate.
// TODO(urvang): Can we pick one of the two heuristically (based on frame
// rectangle and/or presence of alpha)?
static WebPMuxError OptimizeSingleFrame(WebPAnimEncoder* const enc,
                                        WebPData* const webp_data) {
  WebPMuxError err = WEBP_MUX_OK;
  int canvas_width, canvas_height;
  WebPMuxFrameInfo frame;
  WebPData full_image;
  WebPData webp_data2;
  WebPMux* const mux = WebPMuxCreate(webp_data, 0);
  if (mux == NULL) return WEBP_MUX_BAD_DATA;
  assert(enc->out_frame_count == 1);
  WebPDataInit(&frame.bitstream);
  WebPDataInit(&full_image);
  WebPDataInit(&webp_data2);

  err = WebPMuxGetFrame(mux, 1, &frame);
  if (err != WEBP_MUX_OK) goto End;
  if (frame.id != WEBP_CHUNK_ANMF) goto End;  // Non-animation: nothing to do.
  err = WebPMuxGetCanvasSize(mux, &canvas_width, &canvas_height);
  if (err != WEBP_MUX_OK) goto End;
  if (!FrameToFullCanvas(enc, &frame, &full_image)) {
    err = WEBP_MUX_BAD_DATA;
    goto End;
  }
  err = WebPMuxSetImage(mux, &full_image, 1);
  if (err != WEBP_MUX_OK) goto End;
  err = WebPMuxAssemble(mux, &webp_data2);
  if (err != WEBP_MUX_OK) goto End;

  if (webp_data2.size < webp_data->size) {  // Pick 'webp_data2' if smaller.
    WebPDataClear(webp_data);
    *webp_data = webp_data2;
    WebPDataInit(&webp_data2);
  }

 End:
  WebPDataClear(&frame.bitstream);
  WebPDataClear(&full_image);
  WebPMuxDelete(mux);
  WebPDataClear(&webp_data2);
  return err;
}

int WebPAnimEncoderAssemble(WebPAnimEncoder* enc, WebPData* webp_data) {
  WebPMux* mux;
  WebPMuxError err;

  if (enc == NULL) {
    return 0;
  }
  MarkNoError(enc);

  if (webp_data == NULL) {
    MarkError(enc, "ERROR assembling: NULL input");
    return 0;
  }

  if (enc->in_frame_count == 0) {
    MarkError(enc, "ERROR: No frames to assemble");
    return 0;
  }

  if (!enc->got_null_frame && enc->in_frame_count > 1 && enc->count > 0) {
    // set duration of the last frame to be avg of durations of previous frames.
    const double delta_time =
        (uint32_t)enc->prev_timestamp - enc->first_timestamp;
    const int average_duration = (int)(delta_time / (enc->in_frame_count - 1));
    if (!IncreasePreviousDuration(enc, average_duration)) {
      return 0;
    }
  }

  // Flush any remaining frames.
  enc->flush_count = enc->count;
  if (!FlushFrames(enc)) {
    return 0;
  }

  // Set definitive canvas size.
  mux = enc->mux;
  err = WebPMuxSetCanvasSize(mux, enc->canvas_width, enc->canvas_height);
  if (err != WEBP_MUX_OK) goto Err;

  err = WebPMuxSetAnimationParams(mux, &enc->options.anim_params);
  if (err != WEBP_MUX_OK) goto Err;

  // Assemble into a WebP bitstream.
  err = WebPMuxAssemble(mux, webp_data);
  if (err != WEBP_MUX_OK) goto Err;

  if (enc->out_frame_count == 1) {
    err = OptimizeSingleFrame(enc, webp_data);
    if (err != WEBP_MUX_OK) goto Err;
  }
  return 1;

 Err:
  MarkError2(enc, "ERROR assembling WebP", err);
  return 0;
}

const char* WebPAnimEncoderGetError(WebPAnimEncoder* enc) {
  if (enc == NULL) return NULL;
  return enc->error_str;
}

WebPMuxError WebPAnimEncoderSetChunk(
    WebPAnimEncoder* enc, const char fourcc[4], const WebPData* chunk_data,
    int copy_data) {
  if (enc == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return WebPMuxSetChunk(enc->mux, fourcc, chunk_data, copy_data);
}

WebPMuxError WebPAnimEncoderGetChunk(
    const WebPAnimEncoder* enc, const char fourcc[4], WebPData* chunk_data) {
  if (enc == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return WebPMuxGetChunk(enc->mux, fourcc, chunk_data);
}

WebPMuxError WebPAnimEncoderDeleteChunk(
    WebPAnimEncoder* enc, const char fourcc[4]) {
  if (enc == NULL) return WEBP_MUX_INVALID_ARGUMENT;
  return WebPMuxDeleteChunk(enc->mux, fourcc);
}

// -----------------------------------------------------------------------------
