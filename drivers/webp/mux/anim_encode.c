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
#include <stdio.h>

#include "../utils/utils.h"
#include "../webp/decode.h"
#include "../webp/encode.h"
#include "../webp/format_constants.h"
#include "../webp/mux.h"

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

#define ERROR_STR_MAX_LENGTH 100

//------------------------------------------------------------------------------
// Internal structs.

// Stores frame rectangle dimensions.
typedef struct {
  int x_offset_, y_offset_, width_, height_;
} FrameRect;

// Used to store two candidates of encoded data for an animation frame. One of
// the two will be chosen later.
typedef struct {
  WebPMuxFrameInfo sub_frame_;  // Encoded frame rectangle.
  WebPMuxFrameInfo key_frame_;  // Encoded frame if it is a key-frame.
  int is_key_frame_;            // True if 'key_frame' has been chosen.
} EncodedFrame;

struct WebPAnimEncoder {
  const int canvas_width_;                  // Canvas width.
  const int canvas_height_;                 // Canvas height.
  const WebPAnimEncoderOptions options_;    // Global encoding options.

  FrameRect prev_rect_;               // Previous WebP frame rectangle.
  WebPConfig last_config_;            // Cached in case a re-encode is needed.
  WebPConfig last_config2_;           // 2nd cached config; only valid if
                                      // 'options_.allow_mixed' is true.

  WebPPicture* curr_canvas_;          // Only pointer; we don't own memory.

  // Canvas buffers.
  WebPPicture curr_canvas_copy_;      // Possibly modified current canvas.
  int curr_canvas_copy_modified_;     // True if pixels in 'curr_canvas_copy_'
                                      // differ from those in 'curr_canvas_'.

  WebPPicture prev_canvas_;           // Previous canvas.
  WebPPicture prev_canvas_disposed_;  // Previous canvas disposed to background.

  // Encoded data.
  EncodedFrame* encoded_frames_;      // Array of encoded frames.
  size_t size_;             // Number of allocated frames.
  size_t start_;            // Frame start index.
  size_t count_;            // Number of valid frames.
  size_t flush_count_;      // If >0, 'flush_count' frames starting from
                            // 'start' are ready to be added to mux.

  // key-frame related.
  int64_t best_delta_;      // min(canvas size - frame size) over the frames.
                            // Can be negative in certain cases due to
                            // transparent pixels in a frame.
  int keyframe_;            // Index of selected key-frame relative to 'start_'.
  int count_since_key_frame_;     // Frames seen since the last key-frame.

  int first_timestamp_;           // Timestamp of the first frame.
  int prev_timestamp_;            // Timestamp of the last added frame.
  int prev_candidate_undecided_;  // True if it's not yet decided if previous
                                  // frame would be a sub-frame or a key-frame.

  // Misc.
  int is_first_frame_;  // True if first frame is yet to be added/being added.
  int got_null_frame_;  // True if WebPAnimEncoderAdd() has already been called
                        // with a NULL frame.

  size_t in_frame_count_;   // Number of input frames processed so far.
  size_t out_frame_count_;  // Number of frames added to mux so far. This may be
                            // different from 'in_frame_count_' due to merging.

  WebPMux* mux_;        // Muxer to assemble the WebP bitstream.
  char error_str_[ERROR_STR_MAX_LENGTH];  // Error string. Empty if no error.
};

// -----------------------------------------------------------------------------
// Life of WebPAnimEncoder object.

#define DELTA_INFINITY      (1ULL << 32)
#define KEYFRAME_NONE       (-1)

// Reset the counters in the WebPAnimEncoder.
static void ResetCounters(WebPAnimEncoder* const enc) {
  enc->start_ = 0;
  enc->count_ = 0;
  enc->flush_count_ = 0;
  enc->best_delta_ = DELTA_INFINITY;
  enc->keyframe_ = KEYFRAME_NONE;
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

  if (enc_options->kmin <= 0) {
    DisableKeyframes(enc_options);
    print_warning = 0;
  }
  if (enc_options->kmax <= 0) {  // All frames will be key-frames.
    enc_options->kmin = 0;
    enc_options->kmax = 0;
    return;
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

#define TRANSPARENT_COLOR   0x00ffffff

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
                             const FrameRect* const rect) {
  if (rect != NULL) {
    ClearRectangle(picture, rect->x_offset_, rect->y_offset_,
                   rect->width_, rect->height_);
  } else {
    ClearRectangle(picture, 0, 0, picture->width, picture->height);
  }
}

static void MarkNoError(WebPAnimEncoder* const enc) {
  enc->error_str_[0] = '\0';  // Empty string.
}

static void MarkError(WebPAnimEncoder* const enc, const char* str) {
  if (snprintf(enc->error_str_, ERROR_STR_MAX_LENGTH, "%s.", str) < 0) {
    assert(0);  // FIX ME!
  }
}

static void MarkError2(WebPAnimEncoder* const enc,
                       const char* str, int error_code) {
  if (snprintf(enc->error_str_, ERROR_STR_MAX_LENGTH, "%s: %d.", str,
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
  // sanity inits, so we can call WebPAnimEncoderDelete():
  enc->encoded_frames_ = NULL;
  enc->mux_ = NULL;
  MarkNoError(enc);

  // Dimensions and options.
  *(int*)&enc->canvas_width_ = width;
  *(int*)&enc->canvas_height_ = height;
  if (enc_options != NULL) {
    *(WebPAnimEncoderOptions*)&enc->options_ = *enc_options;
    SanitizeEncoderOptions((WebPAnimEncoderOptions*)&enc->options_);
  } else {
    DefaultEncoderOptions((WebPAnimEncoderOptions*)&enc->options_);
  }

  // Canvas buffers.
  if (!WebPPictureInit(&enc->curr_canvas_copy_) ||
      !WebPPictureInit(&enc->prev_canvas_) ||
      !WebPPictureInit(&enc->prev_canvas_disposed_)) {
    goto Err;
  }
  enc->curr_canvas_copy_.width = width;
  enc->curr_canvas_copy_.height = height;
  enc->curr_canvas_copy_.use_argb = 1;
  if (!WebPPictureAlloc(&enc->curr_canvas_copy_) ||
      !WebPPictureCopy(&enc->curr_canvas_copy_, &enc->prev_canvas_) ||
      !WebPPictureCopy(&enc->curr_canvas_copy_, &enc->prev_canvas_disposed_)) {
    goto Err;
  }
  WebPUtilClearPic(&enc->prev_canvas_, NULL);
  enc->curr_canvas_copy_modified_ = 1;

  // Encoded frames.
  ResetCounters(enc);
  // Note: one extra storage is for the previous frame.
  enc->size_ = enc->options_.kmax - enc->options_.kmin + 1;
  // We need space for at least 2 frames. But when kmin, kmax are both zero,
  // enc->size_ will be 1. So we handle that special case below.
  if (enc->size_ < 2) enc->size_ = 2;
  enc->encoded_frames_ =
      (EncodedFrame*)WebPSafeCalloc(enc->size_, sizeof(*enc->encoded_frames_));
  if (enc->encoded_frames_ == NULL) goto Err;

  enc->mux_ = WebPMuxNew();
  if (enc->mux_ == NULL) goto Err;

  enc->count_since_key_frame_ = 0;
  enc->first_timestamp_ = 0;
  enc->prev_timestamp_ = 0;
  enc->prev_candidate_undecided_ = 0;
  enc->is_first_frame_ = 1;
  enc->got_null_frame_ = 0;

  return enc;  // All OK.

 Err:
  WebPAnimEncoderDelete(enc);
  return NULL;
}

// Release the data contained by 'encoded_frame'.
static void FrameRelease(EncodedFrame* const encoded_frame) {
  if (encoded_frame != NULL) {
    WebPDataClear(&encoded_frame->sub_frame_.bitstream);
    WebPDataClear(&encoded_frame->key_frame_.bitstream);
    memset(encoded_frame, 0, sizeof(*encoded_frame));
  }
}

void WebPAnimEncoderDelete(WebPAnimEncoder* enc) {
  if (enc != NULL) {
    WebPPictureFree(&enc->curr_canvas_copy_);
    WebPPictureFree(&enc->prev_canvas_);
    WebPPictureFree(&enc->prev_canvas_disposed_);
    if (enc->encoded_frames_ != NULL) {
      size_t i;
      for (i = 0; i < enc->size_; ++i) {
        FrameRelease(&enc->encoded_frames_[i]);
      }
      WebPSafeFree(enc->encoded_frames_);
    }
    WebPMuxDelete(enc->mux_);
    WebPSafeFree(enc);
  }
}

// -----------------------------------------------------------------------------
// Frame addition.

// Returns cached frame at the given 'position'.
static EncodedFrame* GetFrame(const WebPAnimEncoder* const enc,
                              size_t position) {
  assert(enc->start_ + position < enc->size_);
  return &enc->encoded_frames_[enc->start_ + position];
}

// Returns true if 'length' number of pixels in 'src' and 'dst' are identical,
// assuming the given step sizes between pixels.
static WEBP_INLINE int ComparePixels(const uint32_t* src, int src_step,
                                     const uint32_t* dst, int dst_step,
                                     int length) {
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

static int IsEmptyRect(const FrameRect* const rect) {
  return (rect->width_ == 0) || (rect->height_ == 0);
}

// Assumes that an initial valid guess of change rectangle 'rect' is passed.
static void MinimizeChangeRectangle(const WebPPicture* const src,
                                    const WebPPicture* const dst,
                                    FrameRect* const rect) {
  int i, j;
  // Sanity checks.
  assert(src->width == dst->width && src->height == dst->height);
  assert(rect->x_offset_ + rect->width_ <= dst->width);
  assert(rect->y_offset_ + rect->height_ <= dst->height);

  // Left boundary.
  for (i = rect->x_offset_; i < rect->x_offset_ + rect->width_; ++i) {
    const uint32_t* const src_argb =
        &src->argb[rect->y_offset_ * src->argb_stride + i];
    const uint32_t* const dst_argb =
        &dst->argb[rect->y_offset_ * dst->argb_stride + i];
    if (ComparePixels(src_argb, src->argb_stride, dst_argb, dst->argb_stride,
                      rect->height_)) {
      --rect->width_;  // Redundant column.
      ++rect->x_offset_;
    } else {
      break;
    }
  }
  if (rect->width_ == 0) goto NoChange;

  // Right boundary.
  for (i = rect->x_offset_ + rect->width_ - 1; i >= rect->x_offset_; --i) {
    const uint32_t* const src_argb =
        &src->argb[rect->y_offset_ * src->argb_stride + i];
    const uint32_t* const dst_argb =
        &dst->argb[rect->y_offset_ * dst->argb_stride + i];
    if (ComparePixels(src_argb, src->argb_stride, dst_argb, dst->argb_stride,
                      rect->height_)) {
      --rect->width_;  // Redundant column.
    } else {
      break;
    }
  }
  if (rect->width_ == 0) goto NoChange;

  // Top boundary.
  for (j = rect->y_offset_; j < rect->y_offset_ + rect->height_; ++j) {
    const uint32_t* const src_argb =
        &src->argb[j * src->argb_stride + rect->x_offset_];
    const uint32_t* const dst_argb =
        &dst->argb[j * dst->argb_stride + rect->x_offset_];
    if (ComparePixels(src_argb, 1, dst_argb, 1, rect->width_)) {
      --rect->height_;  // Redundant row.
      ++rect->y_offset_;
    } else {
      break;
    }
  }
  if (rect->height_ == 0) goto NoChange;

  // Bottom boundary.
  for (j = rect->y_offset_ + rect->height_ - 1; j >= rect->y_offset_; --j) {
    const uint32_t* const src_argb =
        &src->argb[j * src->argb_stride + rect->x_offset_];
    const uint32_t* const dst_argb =
        &dst->argb[j * dst->argb_stride + rect->x_offset_];
    if (ComparePixels(src_argb, 1, dst_argb, 1, rect->width_)) {
      --rect->height_;  // Redundant row.
    } else {
      break;
    }
  }
  if (rect->height_ == 0) goto NoChange;

  if (IsEmptyRect(rect)) {
 NoChange:
    rect->x_offset_ = 0;
    rect->y_offset_ = 0;
    rect->width_ = 0;
    rect->height_ = 0;
  }
}

// Snap rectangle to even offsets (and adjust dimensions if needed).
static WEBP_INLINE void SnapToEvenOffsets(FrameRect* const rect) {
  rect->width_ += (rect->x_offset_ & 1);
  rect->height_ += (rect->y_offset_ & 1);
  rect->x_offset_ &= ~1;
  rect->y_offset_ &= ~1;
}

// Given previous and current canvas, picks the optimal rectangle for the
// current frame. The initial guess for 'rect' will be the full canvas.
static int GetSubRect(const WebPPicture* const prev_canvas,
                      const WebPPicture* const curr_canvas, int is_key_frame,
                      int is_first_frame, int empty_rect_allowed,
                      FrameRect* const rect, WebPPicture* const sub_frame) {
  rect->x_offset_ = 0;
  rect->y_offset_ = 0;
  rect->width_ = curr_canvas->width;
  rect->height_ = curr_canvas->height;
  if (!is_key_frame || is_first_frame) {  // Optimize frame rectangle.
    // Note: This behaves as expected for first frame, as 'prev_canvas' is
    // initialized to a fully transparent canvas in the beginning.
    MinimizeChangeRectangle(prev_canvas, curr_canvas, rect);
  }

  if (IsEmptyRect(rect)) {
    if (empty_rect_allowed) {  // No need to get 'sub_frame'.
      return 1;
    } else {                   // Force a 1x1 rectangle.
      rect->width_ = 1;
      rect->height_ = 1;
      assert(rect->x_offset_ == 0);
      assert(rect->y_offset_ == 0);
    }
  }

  SnapToEvenOffsets(rect);
  return WebPPictureView(curr_canvas, rect->x_offset_, rect->y_offset_,
                         rect->width_, rect->height_, sub_frame);
}

static void DisposeFrameRectangle(int dispose_method,
                                  const FrameRect* const rect,
                                  WebPPicture* const curr_canvas) {
  assert(rect != NULL);
  if (dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
    WebPUtilClearPic(curr_canvas, rect);
  }
}

static uint32_t RectArea(const FrameRect* const rect) {
  return (uint32_t)rect->width_ * rect->height_;
}

static int IsBlendingPossible(const WebPPicture* const src,
                              const WebPPicture* const dst,
                              const FrameRect* const rect) {
  int i, j;
  assert(src->width == dst->width && src->height == dst->height);
  assert(rect->x_offset_ + rect->width_ <= dst->width);
  assert(rect->y_offset_ + rect->height_ <= dst->height);
  for (j = rect->y_offset_; j < rect->y_offset_ + rect->height_; ++j) {
    for (i = rect->x_offset_; i < rect->x_offset_ + rect->width_; ++i) {
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

#define MIN_COLORS_LOSSY     31  // Don't try lossy below this threshold.
#define MAX_COLORS_LOSSLESS 194  // Don't try lossless above this threshold.
#define MAX_COLOR_COUNT     256  // Power of 2 greater than MAX_COLORS_LOSSLESS.
#define HASH_SIZE (MAX_COLOR_COUNT * 4)
#define HASH_RIGHT_SHIFT     22  // 32 - log2(HASH_SIZE).

// TODO(urvang): Also used in enc/vp8l.c. Move to utils.
// If the number of colors in the 'pic' is at least MAX_COLOR_COUNT, return
// MAX_COLOR_COUNT. Otherwise, return the exact number of colors in the 'pic'.
static int GetColorCount(const WebPPicture* const pic) {
  int x, y;
  int num_colors = 0;
  uint8_t in_use[HASH_SIZE] = { 0 };
  uint32_t colors[HASH_SIZE];
  static const uint32_t kHashMul = 0x1e35a7bd;
  const uint32_t* argb = pic->argb;
  const int width = pic->width;
  const int height = pic->height;
  uint32_t last_pix = ~argb[0];   // so we're sure that last_pix != argb[0]

  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int key;
      if (argb[x] == last_pix) {
        continue;
      }
      last_pix = argb[x];
      key = (kHashMul * last_pix) >> HASH_RIGHT_SHIFT;
      while (1) {
        if (!in_use[key]) {
          colors[key] = last_pix;
          in_use[key] = 1;
          ++num_colors;
          if (num_colors >= MAX_COLOR_COUNT) {
            return MAX_COLOR_COUNT;  // Exact count not needed.
          }
          break;
        } else if (colors[key] == last_pix) {
          break;  // The color is already there.
        } else {
          // Some other color sits here, so do linear conflict resolution.
          ++key;
          key &= (HASH_SIZE - 1);  // Key mask.
        }
      }
    }
    argb += pic->argb_stride;
  }
  return num_colors;
}

#undef MAX_COLOR_COUNT
#undef HASH_SIZE
#undef HASH_RIGHT_SHIFT

// For pixels in 'rect', replace those pixels in 'dst' that are same as 'src' by
// transparent pixels.
static void IncreaseTransparency(const WebPPicture* const src,
                                 const FrameRect* const rect,
                                 WebPPicture* const dst) {
  int i, j;
  assert(src != NULL && dst != NULL && rect != NULL);
  assert(src->width == dst->width && src->height == dst->height);
  for (j = rect->y_offset_; j < rect->y_offset_ + rect->height_; ++j) {
    const uint32_t* const psrc = src->argb + j * src->argb_stride;
    uint32_t* const pdst = dst->argb + j * dst->argb_stride;
    for (i = rect->x_offset_; i < rect->x_offset_ + rect->width_; ++i) {
      if (psrc[i] == pdst[i]) {
        pdst[i] = TRANSPARENT_COLOR;
      }
    }
  }
}

#undef TRANSPARENT_COLOR

// Replace similar blocks of pixels by a 'see-through' transparent block
// with uniform average color.
static void FlattenSimilarBlocks(const WebPPicture* const src,
                                 const FrameRect* const rect,
                                 WebPPicture* const dst) {
  int i, j;
  const int block_size = 8;
  const int y_start = (rect->y_offset_ + block_size) & ~(block_size - 1);
  const int y_end = (rect->y_offset_ + rect->height_) & ~(block_size - 1);
  const int x_start = (rect->x_offset_ + block_size) & ~(block_size - 1);
  const int x_end = (rect->x_offset_ + rect->width_) & ~(block_size - 1);
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
              src_pixel == pdst[x + y * dst->argb_stride]) {
              ++cnt;
              avg_r += (src_pixel >> 16) & 0xff;
              avg_g += (src_pixel >>  8) & 0xff;
              avg_b += (src_pixel >>  0) & 0xff;
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
      }
    }
  }
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
  WebPMemoryWriter  mem_;
  WebPMuxFrameInfo  info_;
  FrameRect         rect_;
  int               evaluate_;  // True if this candidate should be evaluated.
} Candidate;

// Generates a candidate encoded frame given a picture and metadata.
static WebPEncodingError EncodeCandidate(WebPPicture* const sub_frame,
                                         const FrameRect* const rect,
                                         const WebPConfig* const config,
                                         int use_blending,
                                         Candidate* const candidate) {
  WebPEncodingError error_code = VP8_ENC_OK;
  assert(candidate != NULL);
  memset(candidate, 0, sizeof(*candidate));

  // Set frame rect and info.
  candidate->rect_ = *rect;
  candidate->info_.id = WEBP_CHUNK_ANMF;
  candidate->info_.x_offset = rect->x_offset_;
  candidate->info_.y_offset = rect->y_offset_;
  candidate->info_.dispose_method = WEBP_MUX_DISPOSE_NONE;  // Set later.
  candidate->info_.blend_method =
      use_blending ? WEBP_MUX_BLEND : WEBP_MUX_NO_BLEND;
  candidate->info_.duration = 0;  // Set in next call to WebPAnimEncoderAdd().

  // Encode picture.
  WebPMemoryWriterInit(&candidate->mem_);

  if (!EncodeFrame(config, sub_frame, &candidate->mem_)) {
    error_code = sub_frame->error_code;
    goto Err;
  }

  candidate->evaluate_ = 1;
  return error_code;

 Err:
  WebPMemoryWriterClear(&candidate->mem_);
  return error_code;
}

static void CopyCurrentCanvas(WebPAnimEncoder* const enc) {
  if (enc->curr_canvas_copy_modified_) {
    WebPCopyPixels(enc->curr_canvas_, &enc->curr_canvas_copy_);
    enc->curr_canvas_copy_modified_ = 0;
  }
}

enum {
  LL_DISP_NONE = 0,
  LL_DISP_BG,
  LOSSY_DISP_NONE,
  LOSSY_DISP_BG,
  CANDIDATE_COUNT
};

// Generates candidates for a given dispose method given pre-filled 'rect'
// and 'sub_frame'.
static WebPEncodingError GenerateCandidates(
    WebPAnimEncoder* const enc, Candidate candidates[CANDIDATE_COUNT],
    WebPMuxAnimDispose dispose_method, int is_lossless, int is_key_frame,
    const FrameRect* const rect, WebPPicture* sub_frame,
    const WebPConfig* const config_ll, const WebPConfig* const config_lossy) {
  WebPEncodingError error_code = VP8_ENC_OK;
  const int is_dispose_none = (dispose_method == WEBP_MUX_DISPOSE_NONE);
  Candidate* const candidate_ll =
      is_dispose_none ? &candidates[LL_DISP_NONE] : &candidates[LL_DISP_BG];
  Candidate* const candidate_lossy = is_dispose_none
                                     ? &candidates[LOSSY_DISP_NONE]
                                     : &candidates[LOSSY_DISP_BG];
  WebPPicture* const curr_canvas = &enc->curr_canvas_copy_;
  const WebPPicture* const prev_canvas =
      is_dispose_none ? &enc->prev_canvas_ : &enc->prev_canvas_disposed_;
  const int use_blending =
      !is_key_frame &&
      IsBlendingPossible(prev_canvas, curr_canvas, rect);

  // Pick candidates to be tried.
  if (!enc->options_.allow_mixed) {
    candidate_ll->evaluate_ = is_lossless;
    candidate_lossy->evaluate_ = !is_lossless;
  } else {  // Use a heuristic for trying lossless and/or lossy compression.
    const int num_colors = GetColorCount(sub_frame);
    candidate_ll->evaluate_ = (num_colors < MAX_COLORS_LOSSLESS);
    candidate_lossy->evaluate_ = (num_colors >= MIN_COLORS_LOSSY);
  }

  // Generate candidates.
  if (candidate_ll->evaluate_) {
    CopyCurrentCanvas(enc);
    if (use_blending) {
      IncreaseTransparency(prev_canvas, rect, curr_canvas);
      enc->curr_canvas_copy_modified_ = 1;
    }
    error_code = EncodeCandidate(sub_frame, rect, config_ll, use_blending,
                                 candidate_ll);
    if (error_code != VP8_ENC_OK) return error_code;
  }
  if (candidate_lossy->evaluate_) {
    CopyCurrentCanvas(enc);
    if (use_blending) {
      FlattenSimilarBlocks(prev_canvas, rect, curr_canvas);
      enc->curr_canvas_copy_modified_ = 1;
    }
    error_code = EncodeCandidate(sub_frame, rect, config_lossy, use_blending,
                                 candidate_lossy);
    if (error_code != VP8_ENC_OK) return error_code;
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
  const size_t position = enc->count_ - 2;
  EncodedFrame* const prev_enc_frame = GetFrame(enc, position);
  assert(enc->count_ >= 2);  // As current and previous frames are in enc.

  if (enc->prev_candidate_undecided_) {
    assert(dispose_method == WEBP_MUX_DISPOSE_NONE);
    prev_enc_frame->sub_frame_.dispose_method = dispose_method;
    prev_enc_frame->key_frame_.dispose_method = dispose_method;
  } else {
    WebPMuxFrameInfo* const prev_info = prev_enc_frame->is_key_frame_
                                        ? &prev_enc_frame->key_frame_
                                        : &prev_enc_frame->sub_frame_;
    prev_info->dispose_method = dispose_method;
  }
}

static int IncreasePreviousDuration(WebPAnimEncoder* const enc, int duration) {
  const size_t position = enc->count_ - 1;
  EncodedFrame* const prev_enc_frame = GetFrame(enc, position);
  int new_duration;

  assert(enc->count_ >= 1);
  assert(prev_enc_frame->sub_frame_.duration ==
         prev_enc_frame->key_frame_.duration);
  assert(prev_enc_frame->sub_frame_.duration ==
         (prev_enc_frame->sub_frame_.duration & (MAX_DURATION - 1)));
  assert(duration == (duration & (MAX_DURATION - 1)));

  new_duration = prev_enc_frame->sub_frame_.duration + duration;
  if (new_duration >= MAX_DURATION) {  // Special case.
    // Separate out previous frame from earlier merged frames to avoid overflow.
    // We add a 1x1 transparent frame for the previous frame, with blending on.
    const FrameRect rect = { 0, 0, 1, 1 };
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
        (enc->last_config_.lossless || enc->options_.allow_mixed);
    EncodedFrame* const curr_enc_frame = GetFrame(enc, enc->count_);
    curr_enc_frame->is_key_frame_ = 0;
    curr_enc_frame->sub_frame_.id = WEBP_CHUNK_ANMF;
    curr_enc_frame->sub_frame_.x_offset = 0;
    curr_enc_frame->sub_frame_.y_offset = 0;
    curr_enc_frame->sub_frame_.dispose_method = WEBP_MUX_DISPOSE_NONE;
    curr_enc_frame->sub_frame_.blend_method = WEBP_MUX_BLEND;
    curr_enc_frame->sub_frame_.duration = duration;
    if (!WebPDataCopy(can_use_lossless ? &lossless_1x1 : &lossy_1x1,
                      &curr_enc_frame->sub_frame_.bitstream)) {
      return 0;
    }
    ++enc->count_;
    ++enc->count_since_key_frame_;
    enc->flush_count_ = enc->count_ - 1;
    enc->prev_candidate_undecided_ = 0;
    enc->prev_rect_ = rect;
  } else {                           // Regular case.
    // Increase duration of the previous frame by 'duration'.
    prev_enc_frame->sub_frame_.duration = new_duration;
    prev_enc_frame->key_frame_.duration = new_duration;
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
    if (candidates[i].evaluate_) {
      const size_t candidate_size = candidates[i].mem_.size;
      if (candidate_size < best_size) {
        best_idx = i;
        best_size = candidate_size;
      }
    }
  }
  assert(best_idx != -1);
  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    if (candidates[i].evaluate_) {
      if (i == best_idx) {
        WebPMuxFrameInfo* const dst = is_key_frame
                                      ? &encoded_frame->key_frame_
                                      : &encoded_frame->sub_frame_;
        *dst = candidates[i].info_;
        GetEncodedData(&candidates[i].mem_, &dst->bitstream);
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
        enc->prev_rect_ = candidates[i].rect_;  // save for next frame.
      } else {
        WebPMemoryWriterClear(&candidates[i].mem_);
        candidates[i].evaluate_ = 0;
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
  const WebPPicture* const curr_canvas = &enc->curr_canvas_copy_;
  const WebPPicture* const prev_canvas = &enc->prev_canvas_;
  Candidate candidates[CANDIDATE_COUNT];
  const int is_lossless = config->lossless;
  const int is_first_frame = enc->is_first_frame_;

  int try_dispose_none = 1;  // Default.
  FrameRect rect_none;
  WebPPicture sub_frame_none;
  // First frame cannot be skipped as there is no 'previous frame' to merge it
  // to. So, empty rectangle is not allowed for the first frame.
  const int empty_rect_allowed_none = !is_first_frame;

  // If current frame is a key-frame, dispose method of previous frame doesn't
  // matter, so we don't try dispose to background.
  // Also, if key-frame insertion is on, and previous frame could be picked as
  // either a sub-frame or a key-frame, then we can't be sure about what frame
  // rectangle would be disposed. In that case too, we don't try dispose to
  // background.
  const int dispose_bg_possible =
      !is_key_frame && !enc->prev_candidate_undecided_;
  int try_dispose_bg = 0;  // Default.
  FrameRect rect_bg;
  WebPPicture sub_frame_bg;

  WebPConfig config_ll = *config;
  WebPConfig config_lossy = *config;
  config_ll.lossless = 1;
  config_lossy.lossless = 0;
  enc->last_config_ = *config;
  enc->last_config2_ = config->lossless ? config_lossy : config_ll;
  *frame_skipped = 0;

  if (!WebPPictureInit(&sub_frame_none) || !WebPPictureInit(&sub_frame_bg)) {
    return VP8_ENC_ERROR_INVALID_CONFIGURATION;
  }

  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    candidates[i].evaluate_ = 0;
  }

  // Change-rectangle assuming previous frame was DISPOSE_NONE.
  GetSubRect(prev_canvas, curr_canvas, is_key_frame, is_first_frame,
             empty_rect_allowed_none, &rect_none, &sub_frame_none);

  if (IsEmptyRect(&rect_none)) {
    // Don't encode the frame at all. Instead, the duration of the previous
    // frame will be increased later.
    assert(empty_rect_allowed_none);
    *frame_skipped = 1;
    goto End;
  }

  if (dispose_bg_possible) {
    // Change-rectangle assuming previous frame was DISPOSE_BACKGROUND.
    WebPPicture* const prev_canvas_disposed = &enc->prev_canvas_disposed_;
    WebPCopyPixels(prev_canvas, prev_canvas_disposed);
    DisposeFrameRectangle(WEBP_MUX_DISPOSE_BACKGROUND, &enc->prev_rect_,
                          prev_canvas_disposed);
    // Even if there is exact pixel match between 'disposed previous canvas' and
    // 'current canvas', we can't skip current frame, as there may not be exact
    // pixel match between 'previous canvas' and 'current canvas'. So, we don't
    // allow empty rectangle in this case.
    GetSubRect(prev_canvas_disposed, curr_canvas, is_key_frame, is_first_frame,
               0 /* empty_rect_allowed */, &rect_bg, &sub_frame_bg);
    assert(!IsEmptyRect(&rect_bg));

    if (enc->options_.minimize_size) {  // Try both dispose methods.
      try_dispose_bg = 1;
      try_dispose_none = 1;
    } else if (RectArea(&rect_bg) < RectArea(&rect_none)) {
      try_dispose_bg = 1;  // Pick DISPOSE_BACKGROUND.
      try_dispose_none = 0;
    }
  }

  if (try_dispose_none) {
    error_code = GenerateCandidates(
        enc, candidates, WEBP_MUX_DISPOSE_NONE, is_lossless, is_key_frame,
        &rect_none, &sub_frame_none, &config_ll, &config_lossy);
    if (error_code != VP8_ENC_OK) goto Err;
  }

  if (try_dispose_bg) {
    assert(!enc->is_first_frame_);
    assert(dispose_bg_possible);
    error_code = GenerateCandidates(
        enc, candidates, WEBP_MUX_DISPOSE_BACKGROUND, is_lossless, is_key_frame,
        &rect_bg, &sub_frame_bg, &config_ll, &config_lossy);
    if (error_code != VP8_ENC_OK) goto Err;
  }

  PickBestCandidate(enc, candidates, is_key_frame, encoded_frame);

  goto End;

 Err:
  for (i = 0; i < CANDIDATE_COUNT; ++i) {
    if (candidates[i].evaluate_) {
      WebPMemoryWriterClear(&candidates[i].mem_);
    }
  }

 End:
  WebPPictureFree(&sub_frame_none);
  WebPPictureFree(&sub_frame_bg);
  return error_code;
}

// Calculate the penalty incurred if we encode given frame as a key frame
// instead of a sub-frame.
static int64_t KeyFramePenalty(const EncodedFrame* const encoded_frame) {
  return ((int64_t)encoded_frame->key_frame_.bitstream.size -
          encoded_frame->sub_frame_.bitstream.size);
}

static int CacheFrame(WebPAnimEncoder* const enc,
                      const WebPConfig* const config) {
  int ok = 0;
  int frame_skipped = 0;
  WebPEncodingError error_code = VP8_ENC_OK;
  const size_t position = enc->count_;
  EncodedFrame* const encoded_frame = GetFrame(enc, position);

  ++enc->count_;

  if (enc->is_first_frame_) {  // Add this as a key-frame.
    error_code = SetFrame(enc, config, 1, encoded_frame, &frame_skipped);
    if (error_code != VP8_ENC_OK) goto End;
    assert(frame_skipped == 0);  // First frame can't be skipped, even if empty.
    assert(position == 0 && enc->count_ == 1);
    encoded_frame->is_key_frame_ = 1;
    enc->flush_count_ = 0;
    enc->count_since_key_frame_ = 0;
    enc->prev_candidate_undecided_ = 0;
  } else {
    ++enc->count_since_key_frame_;
    if (enc->count_since_key_frame_ <= enc->options_.kmin) {
      // Add this as a frame rectangle.
      error_code = SetFrame(enc, config, 0, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      if (frame_skipped) goto Skip;
      encoded_frame->is_key_frame_ = 0;
      enc->flush_count_ = enc->count_ - 1;
      enc->prev_candidate_undecided_ = 0;
    } else {
      int64_t curr_delta;

      // Add this as a frame rectangle to enc.
      error_code = SetFrame(enc, config, 0, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      if (frame_skipped) goto Skip;

      // Add this as a key-frame to enc, too.
      error_code = SetFrame(enc, config, 1, encoded_frame, &frame_skipped);
      if (error_code != VP8_ENC_OK) goto End;
      assert(frame_skipped == 0);  // Key-frame cannot be an empty rectangle.

      // Analyze size difference of the two variants.
      curr_delta = KeyFramePenalty(encoded_frame);
      if (curr_delta <= enc->best_delta_) {  // Pick this as the key-frame.
        if (enc->keyframe_ != KEYFRAME_NONE) {
          EncodedFrame* const old_keyframe = GetFrame(enc, enc->keyframe_);
          assert(old_keyframe->is_key_frame_);
          old_keyframe->is_key_frame_ = 0;
        }
        encoded_frame->is_key_frame_ = 1;
        enc->keyframe_ = (int)position;
        enc->best_delta_ = curr_delta;
        enc->flush_count_ = enc->count_ - 1;  // We can flush previous frames.
      } else {
        encoded_frame->is_key_frame_ = 0;
      }
      // Note: We need '>=' below because when kmin and kmax are both zero,
      // count_since_key_frame will always be > kmax.
      if (enc->count_since_key_frame_ >= enc->options_.kmax) {
        enc->flush_count_ = enc->count_ - 1;
        enc->count_since_key_frame_ = 0;
        enc->keyframe_ = KEYFRAME_NONE;
        enc->best_delta_ = DELTA_INFINITY;
      }
      enc->prev_candidate_undecided_ = 1;
    }
  }

  // Update previous to previous and previous canvases for next call.
  WebPCopyPixels(enc->curr_canvas_, &enc->prev_canvas_);
  enc->is_first_frame_ = 0;

 Skip:
  ok = 1;
  ++enc->in_frame_count_;

 End:
  if (!ok || frame_skipped) {
    FrameRelease(encoded_frame);
    // We reset some counters, as the frame addition failed/was skipped.
    --enc->count_;
    if (!enc->is_first_frame_) --enc->count_since_key_frame_;
    if (!ok) {
      MarkError2(enc, "ERROR adding frame. WebPEncodingError", error_code);
    }
  }
  enc->curr_canvas_->error_code = error_code;   // report error_code
  assert(ok || error_code != VP8_ENC_OK);
  return ok;
}

static int FlushFrames(WebPAnimEncoder* const enc) {
  while (enc->flush_count_ > 0) {
    WebPMuxError err;
    EncodedFrame* const curr = GetFrame(enc, 0);
    const WebPMuxFrameInfo* const info =
        curr->is_key_frame_ ? &curr->key_frame_ : &curr->sub_frame_;
    assert(enc->mux_ != NULL);
    err = WebPMuxPushFrame(enc->mux_, info, 1);
    if (err != WEBP_MUX_OK) {
      MarkError2(enc, "ERROR adding frame. WebPMuxError", err);
      return 0;
    }
    if (enc->options_.verbose) {
      fprintf(stderr, "INFO: Added frame. offset:%d,%d dispose:%d blend:%d\n",
              info->x_offset, info->y_offset, info->dispose_method,
              info->blend_method);
    }
    ++enc->out_frame_count_;
    FrameRelease(curr);
    ++enc->start_;
    --enc->flush_count_;
    --enc->count_;
    if (enc->keyframe_ != KEYFRAME_NONE) --enc->keyframe_;
  }

  if (enc->count_ == 1 && enc->start_ != 0) {
    // Move enc->start to index 0.
    const int enc_start_tmp = (int)enc->start_;
    EncodedFrame temp = enc->encoded_frames_[0];
    enc->encoded_frames_[0] = enc->encoded_frames_[enc_start_tmp];
    enc->encoded_frames_[enc_start_tmp] = temp;
    FrameRelease(&enc->encoded_frames_[enc_start_tmp]);
    enc->start_ = 0;
  }
  return 1;
}

#undef DELTA_INFINITY
#undef KEYFRAME_NONE

int WebPAnimEncoderAdd(WebPAnimEncoder* enc, WebPPicture* frame, int timestamp,
                       const WebPConfig* encoder_config) {
  WebPConfig config;

  if (enc == NULL) {
    return 0;
  }
  MarkNoError(enc);

  if (!enc->is_first_frame_) {
    // Make sure timestamps are non-decreasing (integer wrap-around is OK).
    const uint32_t prev_frame_duration =
        (uint32_t)timestamp - enc->prev_timestamp_;
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
  } else {
    enc->first_timestamp_ = timestamp;
  }

  if (frame == NULL) {  // Special: last call.
    enc->got_null_frame_ = 1;
    enc->prev_timestamp_ = timestamp;
    return 1;
  }

  if (frame->width != enc->canvas_width_ ||
      frame->height != enc->canvas_height_) {
    frame->error_code = VP8_ENC_ERROR_INVALID_CONFIGURATION;
    MarkError(enc, "ERROR adding frame: Invalid frame dimensions");
    return 0;
  }

  if (!frame->use_argb) {  // Convert frame from YUV(A) to ARGB.
    if (enc->options_.verbose) {
      fprintf(stderr, "WARNING: Converting frame from YUV(A) to ARGB format; "
              "this incurs a small loss.\n");
    }
    if (!WebPPictureYUVAToARGB(frame)) {
      MarkError(enc, "ERROR converting frame from YUV(A) to ARGB");
      return 0;
    }
  }

  if (encoder_config != NULL) {
    config = *encoder_config;
  } else {
    WebPConfigInit(&config);
    config.lossless = 1;
  }
  assert(enc->curr_canvas_ == NULL);
  enc->curr_canvas_ = frame;  // Store reference.
  assert(enc->curr_canvas_copy_modified_ == 1);
  CopyCurrentCanvas(enc);

  if (!CacheFrame(enc, &config)) {
    return 0;
  }

  if (!FlushFrames(enc)) {
    return 0;
  }
  enc->curr_canvas_ = NULL;
  enc->curr_canvas_copy_modified_ = 1;
  enc->prev_timestamp_ = timestamp;
  return 1;
}

// -----------------------------------------------------------------------------
// Bitstream assembly.

static int DecodeFrameOntoCanvas(const WebPMuxFrameInfo* const frame,
                                 WebPPicture* const canvas) {
  const WebPData* const image = &frame->bitstream;
  WebPPicture sub_image;
  WebPDecoderConfig config;
  WebPInitDecoderConfig(&config);
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
  WebPPicture* const canvas_buf = &enc->curr_canvas_copy_;
  WebPMemoryWriter mem1, mem2;
  WebPMemoryWriterInit(&mem1);
  WebPMemoryWriterInit(&mem2);

  if (!DecodeFrameOntoCanvas(frame, canvas_buf)) goto Err;
  if (!EncodeFrame(&enc->last_config_, canvas_buf, &mem1)) goto Err;
  GetEncodedData(&mem1, full_image);

  if (enc->options_.allow_mixed) {
    if (!EncodeFrame(&enc->last_config_, canvas_buf, &mem2)) goto Err;
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
  assert(enc->out_frame_count_ == 1);
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

  if (enc->in_frame_count_ == 0) {
    MarkError(enc, "ERROR: No frames to assemble");
    return 0;
  }

  if (!enc->got_null_frame_ && enc->in_frame_count_ > 1 && enc->count_ > 0) {
    // set duration of the last frame to be avg of durations of previous frames.
    const double delta_time = enc->prev_timestamp_ - enc->first_timestamp_;
    const int average_duration = (int)(delta_time / (enc->in_frame_count_ - 1));
    if (!IncreasePreviousDuration(enc, average_duration)) {
      return 0;
    }
  }

  // Flush any remaining frames.
  enc->flush_count_ = enc->count_;
  if (!FlushFrames(enc)) {
    return 0;
  }

  // Set definitive canvas size.
  mux = enc->mux_;
  err = WebPMuxSetCanvasSize(mux, enc->canvas_width_, enc->canvas_height_);
  if (err != WEBP_MUX_OK) goto Err;

  err = WebPMuxSetAnimationParams(mux, &enc->options_.anim_params);
  if (err != WEBP_MUX_OK) goto Err;

  // Assemble into a WebP bitstream.
  err = WebPMuxAssemble(mux, webp_data);
  if (err != WEBP_MUX_OK) goto Err;

  if (enc->out_frame_count_ == 1) {
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
  return enc->error_str_;
}

// -----------------------------------------------------------------------------
