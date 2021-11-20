// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//  AnimDecoder implementation.
//

#ifdef HAVE_CONFIG_H
#include "src/webp/config.h"
#endif

#include <assert.h>
#include <string.h>

#include "src/utils/utils.h"
#include "src/webp/decode.h"
#include "src/webp/demux.h"

#define NUM_CHANNELS 4

typedef void (*BlendRowFunc)(uint32_t* const, const uint32_t* const, int);
static void BlendPixelRowNonPremult(uint32_t* const src,
                                    const uint32_t* const dst, int num_pixels);
static void BlendPixelRowPremult(uint32_t* const src, const uint32_t* const dst,
                                 int num_pixels);

struct WebPAnimDecoder {
  WebPDemuxer* demux_;             // Demuxer created from given WebP bitstream.
  WebPDecoderConfig config_;       // Decoder config.
  // Note: we use a pointer to a function blending multiple pixels at a time to
  // allow possible inlining of per-pixel blending function.
  BlendRowFunc blend_func_;        // Pointer to the chose blend row function.
  WebPAnimInfo info_;              // Global info about the animation.
  uint8_t* curr_frame_;            // Current canvas (not disposed).
  uint8_t* prev_frame_disposed_;   // Previous canvas (properly disposed).
  int prev_frame_timestamp_;       // Previous frame timestamp (milliseconds).
  WebPIterator prev_iter_;         // Iterator object for previous frame.
  int prev_frame_was_keyframe_;    // True if previous frame was a keyframe.
  int next_frame_;                 // Index of the next frame to be decoded
                                   // (starting from 1).
};

static void DefaultDecoderOptions(WebPAnimDecoderOptions* const dec_options) {
  dec_options->color_mode = MODE_RGBA;
  dec_options->use_threads = 0;
}

int WebPAnimDecoderOptionsInitInternal(WebPAnimDecoderOptions* dec_options,
                                       int abi_version) {
  if (dec_options == NULL ||
      WEBP_ABI_IS_INCOMPATIBLE(abi_version, WEBP_DEMUX_ABI_VERSION)) {
    return 0;
  }
  DefaultDecoderOptions(dec_options);
  return 1;
}

static int ApplyDecoderOptions(const WebPAnimDecoderOptions* const dec_options,
                               WebPAnimDecoder* const dec) {
  WEBP_CSP_MODE mode;
  WebPDecoderConfig* config = &dec->config_;
  assert(dec_options != NULL);

  mode = dec_options->color_mode;
  if (mode != MODE_RGBA && mode != MODE_BGRA &&
      mode != MODE_rgbA && mode != MODE_bgrA) {
    return 0;
  }
  dec->blend_func_ = (mode == MODE_RGBA || mode == MODE_BGRA)
                         ? &BlendPixelRowNonPremult
                         : &BlendPixelRowPremult;
  WebPInitDecoderConfig(config);
  config->output.colorspace = mode;
  config->output.is_external_memory = 1;
  config->options.use_threads = dec_options->use_threads;
  // Note: config->output.u.RGBA is set at the time of decoding each frame.
  return 1;
}

WebPAnimDecoder* WebPAnimDecoderNewInternal(
    const WebPData* webp_data, const WebPAnimDecoderOptions* dec_options,
    int abi_version) {
  WebPAnimDecoderOptions options;
  WebPAnimDecoder* dec = NULL;
  WebPBitstreamFeatures features;
  if (webp_data == NULL ||
      WEBP_ABI_IS_INCOMPATIBLE(abi_version, WEBP_DEMUX_ABI_VERSION)) {
    return NULL;
  }

  // Validate the bitstream before doing expensive allocations. The demuxer may
  // be more tolerant than the decoder.
  if (WebPGetFeatures(webp_data->bytes, webp_data->size, &features) !=
      VP8_STATUS_OK) {
    return NULL;
  }

  // Note: calloc() so that the pointer members are initialized to NULL.
  dec = (WebPAnimDecoder*)WebPSafeCalloc(1ULL, sizeof(*dec));
  if (dec == NULL) goto Error;

  if (dec_options != NULL) {
    options = *dec_options;
  } else {
    DefaultDecoderOptions(&options);
  }
  if (!ApplyDecoderOptions(&options, dec)) goto Error;

  dec->demux_ = WebPDemux(webp_data);
  if (dec->demux_ == NULL) goto Error;

  dec->info_.canvas_width = WebPDemuxGetI(dec->demux_, WEBP_FF_CANVAS_WIDTH);
  dec->info_.canvas_height = WebPDemuxGetI(dec->demux_, WEBP_FF_CANVAS_HEIGHT);
  dec->info_.loop_count = WebPDemuxGetI(dec->demux_, WEBP_FF_LOOP_COUNT);
  dec->info_.bgcolor = WebPDemuxGetI(dec->demux_, WEBP_FF_BACKGROUND_COLOR);
  dec->info_.frame_count = WebPDemuxGetI(dec->demux_, WEBP_FF_FRAME_COUNT);

  // Note: calloc() because we fill frame with zeroes as well.
  dec->curr_frame_ = (uint8_t*)WebPSafeCalloc(
      dec->info_.canvas_width * NUM_CHANNELS, dec->info_.canvas_height);
  if (dec->curr_frame_ == NULL) goto Error;
  dec->prev_frame_disposed_ = (uint8_t*)WebPSafeCalloc(
      dec->info_.canvas_width * NUM_CHANNELS, dec->info_.canvas_height);
  if (dec->prev_frame_disposed_ == NULL) goto Error;

  WebPAnimDecoderReset(dec);
  return dec;

 Error:
  WebPAnimDecoderDelete(dec);
  return NULL;
}

int WebPAnimDecoderGetInfo(const WebPAnimDecoder* dec, WebPAnimInfo* info) {
  if (dec == NULL || info == NULL) return 0;
  *info = dec->info_;
  return 1;
}

// Returns true if the frame covers the full canvas.
static int IsFullFrame(int width, int height, int canvas_width,
                       int canvas_height) {
  return (width == canvas_width && height == canvas_height);
}

// Clear the canvas to transparent.
static int ZeroFillCanvas(uint8_t* buf, uint32_t canvas_width,
                          uint32_t canvas_height) {
  const uint64_t size =
      (uint64_t)canvas_width * canvas_height * NUM_CHANNELS * sizeof(*buf);
  if (!CheckSizeOverflow(size)) return 0;
  memset(buf, 0, (size_t)size);
  return 1;
}

// Clear given frame rectangle to transparent.
static void ZeroFillFrameRect(uint8_t* buf, int buf_stride, int x_offset,
                              int y_offset, int width, int height) {
  int j;
  assert(width * NUM_CHANNELS <= buf_stride);
  buf += y_offset * buf_stride + x_offset * NUM_CHANNELS;
  for (j = 0; j < height; ++j) {
    memset(buf, 0, width * NUM_CHANNELS);
    buf += buf_stride;
  }
}

// Copy width * height pixels from 'src' to 'dst'.
static int CopyCanvas(const uint8_t* src, uint8_t* dst,
                      uint32_t width, uint32_t height) {
  const uint64_t size = (uint64_t)width * height * NUM_CHANNELS;
  if (!CheckSizeOverflow(size)) return 0;
  assert(src != NULL && dst != NULL);
  memcpy(dst, src, (size_t)size);
  return 1;
}

// Returns true if the current frame is a key-frame.
static int IsKeyFrame(const WebPIterator* const curr,
                      const WebPIterator* const prev,
                      int prev_frame_was_key_frame,
                      int canvas_width, int canvas_height) {
  if (curr->frame_num == 1) {
    return 1;
  } else if ((!curr->has_alpha || curr->blend_method == WEBP_MUX_NO_BLEND) &&
             IsFullFrame(curr->width, curr->height,
                         canvas_width, canvas_height)) {
    return 1;
  } else {
    return (prev->dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) &&
           (IsFullFrame(prev->width, prev->height, canvas_width,
                        canvas_height) ||
            prev_frame_was_key_frame);
  }
}


// Blend a single channel of 'src' over 'dst', given their alpha channel values.
// 'src' and 'dst' are assumed to be NOT pre-multiplied by alpha.
static uint8_t BlendChannelNonPremult(uint32_t src, uint8_t src_a,
                                      uint32_t dst, uint8_t dst_a,
                                      uint32_t scale, int shift) {
  const uint8_t src_channel = (src >> shift) & 0xff;
  const uint8_t dst_channel = (dst >> shift) & 0xff;
  const uint32_t blend_unscaled = src_channel * src_a + dst_channel * dst_a;
  assert(blend_unscaled < (1ULL << 32) / scale);
  return (blend_unscaled * scale) >> 24;
}

// Blend 'src' over 'dst' assuming they are NOT pre-multiplied by alpha.
static uint32_t BlendPixelNonPremult(uint32_t src, uint32_t dst) {
  const uint8_t src_a = (src >> 24) & 0xff;

  if (src_a == 0) {
    return dst;
  } else {
    const uint8_t dst_a = (dst >> 24) & 0xff;
    // This is the approximate integer arithmetic for the actual formula:
    // dst_factor_a = (dst_a * (255 - src_a)) / 255.
    const uint8_t dst_factor_a = (dst_a * (256 - src_a)) >> 8;
    const uint8_t blend_a = src_a + dst_factor_a;
    const uint32_t scale = (1UL << 24) / blend_a;

    const uint8_t blend_r =
        BlendChannelNonPremult(src, src_a, dst, dst_factor_a, scale, 0);
    const uint8_t blend_g =
        BlendChannelNonPremult(src, src_a, dst, dst_factor_a, scale, 8);
    const uint8_t blend_b =
        BlendChannelNonPremult(src, src_a, dst, dst_factor_a, scale, 16);
    assert(src_a + dst_factor_a < 256);

    return (blend_r << 0) |
           (blend_g << 8) |
           (blend_b << 16) |
           ((uint32_t)blend_a << 24);
  }
}

// Blend 'num_pixels' in 'src' over 'dst' assuming they are NOT pre-multiplied
// by alpha.
static void BlendPixelRowNonPremult(uint32_t* const src,
                                    const uint32_t* const dst, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint8_t src_alpha = (src[i] >> 24) & 0xff;
    if (src_alpha != 0xff) {
      src[i] = BlendPixelNonPremult(src[i], dst[i]);
    }
  }
}

// Individually multiply each channel in 'pix' by 'scale'.
static WEBP_INLINE uint32_t ChannelwiseMultiply(uint32_t pix, uint32_t scale) {
  uint32_t mask = 0x00FF00FF;
  uint32_t rb = ((pix & mask) * scale) >> 8;
  uint32_t ag = ((pix >> 8) & mask) * scale;
  return (rb & mask) | (ag & ~mask);
}

// Blend 'src' over 'dst' assuming they are pre-multiplied by alpha.
static uint32_t BlendPixelPremult(uint32_t src, uint32_t dst) {
  const uint8_t src_a = (src >> 24) & 0xff;
  return src + ChannelwiseMultiply(dst, 256 - src_a);
}

// Blend 'num_pixels' in 'src' over 'dst' assuming they are pre-multiplied by
// alpha.
static void BlendPixelRowPremult(uint32_t* const src, const uint32_t* const dst,
                                 int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint8_t src_alpha = (src[i] >> 24) & 0xff;
    if (src_alpha != 0xff) {
      src[i] = BlendPixelPremult(src[i], dst[i]);
    }
  }
}

// Returns two ranges (<left, width> pairs) at row 'canvas_y', that belong to
// 'src' but not 'dst'. A point range is empty if the corresponding width is 0.
static void FindBlendRangeAtRow(const WebPIterator* const src,
                                const WebPIterator* const dst, int canvas_y,
                                int* const left1, int* const width1,
                                int* const left2, int* const width2) {
  const int src_max_x = src->x_offset + src->width;
  const int dst_max_x = dst->x_offset + dst->width;
  const int dst_max_y = dst->y_offset + dst->height;
  assert(canvas_y >= src->y_offset && canvas_y < (src->y_offset + src->height));
  *left1 = -1;
  *width1 = 0;
  *left2 = -1;
  *width2 = 0;

  if (canvas_y < dst->y_offset || canvas_y >= dst_max_y ||
      src->x_offset >= dst_max_x || src_max_x <= dst->x_offset) {
    *left1 = src->x_offset;
    *width1 = src->width;
    return;
  }

  if (src->x_offset < dst->x_offset) {
    *left1 = src->x_offset;
    *width1 = dst->x_offset - src->x_offset;
  }

  if (src_max_x > dst_max_x) {
    *left2 = dst_max_x;
    *width2 = src_max_x - dst_max_x;
  }
}

int WebPAnimDecoderGetNext(WebPAnimDecoder* dec,
                           uint8_t** buf_ptr, int* timestamp_ptr) {
  WebPIterator iter;
  uint32_t width;
  uint32_t height;
  int is_key_frame;
  int timestamp;
  BlendRowFunc blend_row;

  if (dec == NULL || buf_ptr == NULL || timestamp_ptr == NULL) return 0;
  if (!WebPAnimDecoderHasMoreFrames(dec)) return 0;

  width = dec->info_.canvas_width;
  height = dec->info_.canvas_height;
  blend_row = dec->blend_func_;

  // Get compressed frame.
  if (!WebPDemuxGetFrame(dec->demux_, dec->next_frame_, &iter)) {
    return 0;
  }
  timestamp = dec->prev_frame_timestamp_ + iter.duration;

  // Initialize.
  is_key_frame = IsKeyFrame(&iter, &dec->prev_iter_,
                            dec->prev_frame_was_keyframe_, width, height);
  if (is_key_frame) {
    if (!ZeroFillCanvas(dec->curr_frame_, width, height)) {
      goto Error;
    }
  } else {
    if (!CopyCanvas(dec->prev_frame_disposed_, dec->curr_frame_,
                    width, height)) {
      goto Error;
    }
  }

  // Decode.
  {
    const uint8_t* in = iter.fragment.bytes;
    const size_t in_size = iter.fragment.size;
    const uint32_t stride = width * NUM_CHANNELS;  // at most 25 + 2 bits
    const uint64_t out_offset = (uint64_t)iter.y_offset * stride +
                                (uint64_t)iter.x_offset * NUM_CHANNELS;  // 53b
    const uint64_t size = (uint64_t)iter.height * stride;  // at most 25 + 27b
    WebPDecoderConfig* const config = &dec->config_;
    WebPRGBABuffer* const buf = &config->output.u.RGBA;
    if ((size_t)size != size) goto Error;
    buf->stride = (int)stride;
    buf->size = (size_t)size;
    buf->rgba = dec->curr_frame_ + out_offset;

    if (WebPDecode(in, in_size, config) != VP8_STATUS_OK) {
      goto Error;
    }
  }

  // During the decoding of current frame, we may have set some pixels to be
  // transparent (i.e. alpha < 255). However, the value of each of these
  // pixels should have been determined by blending it against the value of
  // that pixel in the previous frame if blending method of is WEBP_MUX_BLEND.
  if (iter.frame_num > 1 && iter.blend_method == WEBP_MUX_BLEND &&
      !is_key_frame) {
    if (dec->prev_iter_.dispose_method == WEBP_MUX_DISPOSE_NONE) {
      int y;
      // Blend transparent pixels with pixels in previous canvas.
      for (y = 0; y < iter.height; ++y) {
        const size_t offset =
            (iter.y_offset + y) * width + iter.x_offset;
        blend_row((uint32_t*)dec->curr_frame_ + offset,
                  (uint32_t*)dec->prev_frame_disposed_ + offset, iter.width);
      }
    } else {
      int y;
      assert(dec->prev_iter_.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND);
      // We need to blend a transparent pixel with its value just after
      // initialization. That is, blend it with:
      // * Fully transparent pixel if it belongs to prevRect <-- No-op.
      // * The pixel in the previous canvas otherwise <-- Need alpha-blending.
      for (y = 0; y < iter.height; ++y) {
        const int canvas_y = iter.y_offset + y;
        int left1, width1, left2, width2;
        FindBlendRangeAtRow(&iter, &dec->prev_iter_, canvas_y, &left1, &width1,
                            &left2, &width2);
        if (width1 > 0) {
          const size_t offset1 = canvas_y * width + left1;
          blend_row((uint32_t*)dec->curr_frame_ + offset1,
                    (uint32_t*)dec->prev_frame_disposed_ + offset1, width1);
        }
        if (width2 > 0) {
          const size_t offset2 = canvas_y * width + left2;
          blend_row((uint32_t*)dec->curr_frame_ + offset2,
                    (uint32_t*)dec->prev_frame_disposed_ + offset2, width2);
        }
      }
    }
  }

  // Update info of the previous frame and dispose it for the next iteration.
  dec->prev_frame_timestamp_ = timestamp;
  WebPDemuxReleaseIterator(&dec->prev_iter_);
  dec->prev_iter_ = iter;
  dec->prev_frame_was_keyframe_ = is_key_frame;
  CopyCanvas(dec->curr_frame_, dec->prev_frame_disposed_, width, height);
  if (dec->prev_iter_.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
    ZeroFillFrameRect(dec->prev_frame_disposed_, width * NUM_CHANNELS,
                      dec->prev_iter_.x_offset, dec->prev_iter_.y_offset,
                      dec->prev_iter_.width, dec->prev_iter_.height);
  }
  ++dec->next_frame_;

  // All OK, fill in the values.
  *buf_ptr = dec->curr_frame_;
  *timestamp_ptr = timestamp;
  return 1;

 Error:
  WebPDemuxReleaseIterator(&iter);
  return 0;
}

int WebPAnimDecoderHasMoreFrames(const WebPAnimDecoder* dec) {
  if (dec == NULL) return 0;
  return (dec->next_frame_ <= (int)dec->info_.frame_count);
}

void WebPAnimDecoderReset(WebPAnimDecoder* dec) {
  if (dec != NULL) {
    dec->prev_frame_timestamp_ = 0;
    WebPDemuxReleaseIterator(&dec->prev_iter_);
    memset(&dec->prev_iter_, 0, sizeof(dec->prev_iter_));
    dec->prev_frame_was_keyframe_ = 0;
    dec->next_frame_ = 1;
  }
}

const WebPDemuxer* WebPAnimDecoderGetDemuxer(const WebPAnimDecoder* dec) {
  if (dec == NULL) return NULL;
  return dec->demux_;
}

void WebPAnimDecoderDelete(WebPAnimDecoder* dec) {
  if (dec != NULL) {
    WebPDemuxReleaseIterator(&dec->prev_iter_);
    WebPDemuxDelete(dec->demux_);
    WebPSafeFree(dec->curr_frame_);
    WebPSafeFree(dec->prev_frame_disposed_);
    WebPSafeFree(dec);
  }
}
