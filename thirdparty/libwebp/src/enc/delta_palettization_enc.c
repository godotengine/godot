// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Mislav Bradac (mislavm@google.com)
//

#include "src/enc/delta_palettization_enc.h"

#ifdef WEBP_EXPERIMENTAL_FEATURES
#include "src/webp/types.h"
#include "src/dsp/lossless.h"

#define MK_COL(r, g, b) (((r) << 16) + ((g) << 8) + (b))

// Format allows palette up to 256 entries, but more palette entries produce
// bigger entropy. In the future it will probably be useful to add more entries
// that are far from the origin of the palette or choose remaining entries
// dynamically.
#define DELTA_PALETTE_SIZE 226

// Palette used for delta_palettization. Entries are roughly sorted by distance
// of their signed equivalents from the origin.
static const uint32_t kDeltaPalette[DELTA_PALETTE_SIZE] = {
  MK_COL(0u, 0u, 0u),
  MK_COL(255u, 255u, 255u),
  MK_COL(1u, 1u, 1u),
  MK_COL(254u, 254u, 254u),
  MK_COL(2u, 2u, 2u),
  MK_COL(4u, 4u, 4u),
  MK_COL(252u, 252u, 252u),
  MK_COL(250u, 0u, 0u),
  MK_COL(0u, 250u, 0u),
  MK_COL(0u, 0u, 250u),
  MK_COL(6u, 0u, 0u),
  MK_COL(0u, 6u, 0u),
  MK_COL(0u, 0u, 6u),
  MK_COL(0u, 0u, 248u),
  MK_COL(0u, 0u, 8u),
  MK_COL(0u, 248u, 0u),
  MK_COL(0u, 248u, 248u),
  MK_COL(0u, 248u, 8u),
  MK_COL(0u, 8u, 0u),
  MK_COL(0u, 8u, 248u),
  MK_COL(0u, 8u, 8u),
  MK_COL(8u, 8u, 8u),
  MK_COL(248u, 0u, 0u),
  MK_COL(248u, 0u, 248u),
  MK_COL(248u, 0u, 8u),
  MK_COL(248u, 248u, 0u),
  MK_COL(248u, 8u, 0u),
  MK_COL(8u, 0u, 0u),
  MK_COL(8u, 0u, 248u),
  MK_COL(8u, 0u, 8u),
  MK_COL(8u, 248u, 0u),
  MK_COL(8u, 8u, 0u),
  MK_COL(23u, 23u, 23u),
  MK_COL(13u, 13u, 13u),
  MK_COL(232u, 232u, 232u),
  MK_COL(244u, 244u, 244u),
  MK_COL(245u, 245u, 250u),
  MK_COL(50u, 50u, 50u),
  MK_COL(204u, 204u, 204u),
  MK_COL(236u, 236u, 236u),
  MK_COL(16u, 16u, 16u),
  MK_COL(240u, 16u, 16u),
  MK_COL(16u, 240u, 16u),
  MK_COL(240u, 240u, 16u),
  MK_COL(16u, 16u, 240u),
  MK_COL(240u, 16u, 240u),
  MK_COL(16u, 240u, 240u),
  MK_COL(240u, 240u, 240u),
  MK_COL(0u, 0u, 232u),
  MK_COL(0u, 232u, 0u),
  MK_COL(232u, 0u, 0u),
  MK_COL(0u, 0u, 24u),
  MK_COL(0u, 24u, 0u),
  MK_COL(24u, 0u, 0u),
  MK_COL(32u, 32u, 32u),
  MK_COL(224u, 32u, 32u),
  MK_COL(32u, 224u, 32u),
  MK_COL(224u, 224u, 32u),
  MK_COL(32u, 32u, 224u),
  MK_COL(224u, 32u, 224u),
  MK_COL(32u, 224u, 224u),
  MK_COL(224u, 224u, 224u),
  MK_COL(0u, 0u, 176u),
  MK_COL(0u, 0u, 80u),
  MK_COL(0u, 176u, 0u),
  MK_COL(0u, 176u, 176u),
  MK_COL(0u, 176u, 80u),
  MK_COL(0u, 80u, 0u),
  MK_COL(0u, 80u, 176u),
  MK_COL(0u, 80u, 80u),
  MK_COL(176u, 0u, 0u),
  MK_COL(176u, 0u, 176u),
  MK_COL(176u, 0u, 80u),
  MK_COL(176u, 176u, 0u),
  MK_COL(176u, 80u, 0u),
  MK_COL(80u, 0u, 0u),
  MK_COL(80u, 0u, 176u),
  MK_COL(80u, 0u, 80u),
  MK_COL(80u, 176u, 0u),
  MK_COL(80u, 80u, 0u),
  MK_COL(0u, 0u, 152u),
  MK_COL(0u, 0u, 104u),
  MK_COL(0u, 152u, 0u),
  MK_COL(0u, 152u, 152u),
  MK_COL(0u, 152u, 104u),
  MK_COL(0u, 104u, 0u),
  MK_COL(0u, 104u, 152u),
  MK_COL(0u, 104u, 104u),
  MK_COL(152u, 0u, 0u),
  MK_COL(152u, 0u, 152u),
  MK_COL(152u, 0u, 104u),
  MK_COL(152u, 152u, 0u),
  MK_COL(152u, 104u, 0u),
  MK_COL(104u, 0u, 0u),
  MK_COL(104u, 0u, 152u),
  MK_COL(104u, 0u, 104u),
  MK_COL(104u, 152u, 0u),
  MK_COL(104u, 104u, 0u),
  MK_COL(216u, 216u, 216u),
  MK_COL(216u, 216u, 40u),
  MK_COL(216u, 216u, 176u),
  MK_COL(216u, 216u, 80u),
  MK_COL(216u, 40u, 216u),
  MK_COL(216u, 40u, 40u),
  MK_COL(216u, 40u, 176u),
  MK_COL(216u, 40u, 80u),
  MK_COL(216u, 176u, 216u),
  MK_COL(216u, 176u, 40u),
  MK_COL(216u, 176u, 176u),
  MK_COL(216u, 176u, 80u),
  MK_COL(216u, 80u, 216u),
  MK_COL(216u, 80u, 40u),
  MK_COL(216u, 80u, 176u),
  MK_COL(216u, 80u, 80u),
  MK_COL(40u, 216u, 216u),
  MK_COL(40u, 216u, 40u),
  MK_COL(40u, 216u, 176u),
  MK_COL(40u, 216u, 80u),
  MK_COL(40u, 40u, 216u),
  MK_COL(40u, 40u, 40u),
  MK_COL(40u, 40u, 176u),
  MK_COL(40u, 40u, 80u),
  MK_COL(40u, 176u, 216u),
  MK_COL(40u, 176u, 40u),
  MK_COL(40u, 176u, 176u),
  MK_COL(40u, 176u, 80u),
  MK_COL(40u, 80u, 216u),
  MK_COL(40u, 80u, 40u),
  MK_COL(40u, 80u, 176u),
  MK_COL(40u, 80u, 80u),
  MK_COL(80u, 216u, 216u),
  MK_COL(80u, 216u, 40u),
  MK_COL(80u, 216u, 176u),
  MK_COL(80u, 216u, 80u),
  MK_COL(80u, 40u, 216u),
  MK_COL(80u, 40u, 40u),
  MK_COL(80u, 40u, 176u),
  MK_COL(80u, 40u, 80u),
  MK_COL(80u, 176u, 216u),
  MK_COL(80u, 176u, 40u),
  MK_COL(80u, 176u, 176u),
  MK_COL(80u, 176u, 80u),
  MK_COL(80u, 80u, 216u),
  MK_COL(80u, 80u, 40u),
  MK_COL(80u, 80u, 176u),
  MK_COL(80u, 80u, 80u),
  MK_COL(0u, 0u, 192u),
  MK_COL(0u, 0u, 64u),
  MK_COL(0u, 0u, 128u),
  MK_COL(0u, 192u, 0u),
  MK_COL(0u, 192u, 192u),
  MK_COL(0u, 192u, 64u),
  MK_COL(0u, 192u, 128u),
  MK_COL(0u, 64u, 0u),
  MK_COL(0u, 64u, 192u),
  MK_COL(0u, 64u, 64u),
  MK_COL(0u, 64u, 128u),
  MK_COL(0u, 128u, 0u),
  MK_COL(0u, 128u, 192u),
  MK_COL(0u, 128u, 64u),
  MK_COL(0u, 128u, 128u),
  MK_COL(176u, 216u, 216u),
  MK_COL(176u, 216u, 40u),
  MK_COL(176u, 216u, 176u),
  MK_COL(176u, 216u, 80u),
  MK_COL(176u, 40u, 216u),
  MK_COL(176u, 40u, 40u),
  MK_COL(176u, 40u, 176u),
  MK_COL(176u, 40u, 80u),
  MK_COL(176u, 176u, 216u),
  MK_COL(176u, 176u, 40u),
  MK_COL(176u, 176u, 176u),
  MK_COL(176u, 176u, 80u),
  MK_COL(176u, 80u, 216u),
  MK_COL(176u, 80u, 40u),
  MK_COL(176u, 80u, 176u),
  MK_COL(176u, 80u, 80u),
  MK_COL(192u, 0u, 0u),
  MK_COL(192u, 0u, 192u),
  MK_COL(192u, 0u, 64u),
  MK_COL(192u, 0u, 128u),
  MK_COL(192u, 192u, 0u),
  MK_COL(192u, 192u, 192u),
  MK_COL(192u, 192u, 64u),
  MK_COL(192u, 192u, 128u),
  MK_COL(192u, 64u, 0u),
  MK_COL(192u, 64u, 192u),
  MK_COL(192u, 64u, 64u),
  MK_COL(192u, 64u, 128u),
  MK_COL(192u, 128u, 0u),
  MK_COL(192u, 128u, 192u),
  MK_COL(192u, 128u, 64u),
  MK_COL(192u, 128u, 128u),
  MK_COL(64u, 0u, 0u),
  MK_COL(64u, 0u, 192u),
  MK_COL(64u, 0u, 64u),
  MK_COL(64u, 0u, 128u),
  MK_COL(64u, 192u, 0u),
  MK_COL(64u, 192u, 192u),
  MK_COL(64u, 192u, 64u),
  MK_COL(64u, 192u, 128u),
  MK_COL(64u, 64u, 0u),
  MK_COL(64u, 64u, 192u),
  MK_COL(64u, 64u, 64u),
  MK_COL(64u, 64u, 128u),
  MK_COL(64u, 128u, 0u),
  MK_COL(64u, 128u, 192u),
  MK_COL(64u, 128u, 64u),
  MK_COL(64u, 128u, 128u),
  MK_COL(128u, 0u, 0u),
  MK_COL(128u, 0u, 192u),
  MK_COL(128u, 0u, 64u),
  MK_COL(128u, 0u, 128u),
  MK_COL(128u, 192u, 0u),
  MK_COL(128u, 192u, 192u),
  MK_COL(128u, 192u, 64u),
  MK_COL(128u, 192u, 128u),
  MK_COL(128u, 64u, 0u),
  MK_COL(128u, 64u, 192u),
  MK_COL(128u, 64u, 64u),
  MK_COL(128u, 64u, 128u),
  MK_COL(128u, 128u, 0u),
  MK_COL(128u, 128u, 192u),
  MK_COL(128u, 128u, 64u),
  MK_COL(128u, 128u, 128u),
};

#undef MK_COL

//------------------------------------------------------------------------------
// TODO(skal): move the functions to dsp/lossless.c when the correct
// granularity is found. For now, we'll just copy-paste some useful bits
// here instead.

// In-place sum of each component with mod 256.
static WEBP_INLINE void AddPixelsEq(uint32_t* a, uint32_t b) {
  const uint32_t alpha_and_green = (*a & 0xff00ff00u) + (b & 0xff00ff00u);
  const uint32_t red_and_blue = (*a & 0x00ff00ffu) + (b & 0x00ff00ffu);
  *a = (alpha_and_green & 0xff00ff00u) | (red_and_blue & 0x00ff00ffu);
}

static WEBP_INLINE uint32_t Clip255(uint32_t a) {
  if (a < 256) {
    return a;
  }
  // return 0, when a is a negative integer.
  // return 255, when a is positive.
  return ~a >> 24;
}

// Delta palettization functions.
static WEBP_INLINE int Square(int x) {
  return x * x;
}

static WEBP_INLINE uint32_t Intensity(uint32_t a) {
  return
      30 * ((a >> 16) & 0xff) +
      59 * ((a >>  8) & 0xff) +
      11 * ((a >>  0) & 0xff);
}

static uint32_t CalcDist(uint32_t predicted_value, uint32_t actual_value,
                         uint32_t palette_entry) {
  int i;
  uint32_t distance = 0;
  AddPixelsEq(&predicted_value, palette_entry);
  for (i = 0; i < 32; i += 8) {
    const int32_t av = (actual_value >> i) & 0xff;
    const int32_t pv = (predicted_value >> i) & 0xff;
    distance += Square(pv - av);
  }
  // We sum square of intensity difference with factor 10, but because Intensity
  // returns 100 times real intensity we need to multiply differences of colors
  // by 1000.
  distance *= 1000u;
  distance += Square(Intensity(predicted_value)
                     - Intensity(actual_value));
  return distance;
}

static uint32_t Predict(int x, int y, uint32_t* image) {
  const uint32_t t = (y == 0) ? ARGB_BLACK : image[x];
  const uint32_t l = (x == 0) ? ARGB_BLACK : image[x - 1];
  const uint32_t p =
      (((((t >> 24) & 0xff) + ((l >> 24) & 0xff)) / 2) << 24) +
      (((((t >> 16) & 0xff) + ((l >> 16) & 0xff)) / 2) << 16) +
      (((((t >>  8) & 0xff) + ((l >>  8) & 0xff)) / 2) <<  8) +
      (((((t >>  0) & 0xff) + ((l >>  0) & 0xff)) / 2) <<  0);
  if (x == 0 && y == 0) return ARGB_BLACK;
  if (x == 0) return t;
  if (y == 0) return l;
  return p;
}

static WEBP_INLINE int AddSubtractComponentFullWithCoefficient(
    int a, int b, int c) {
  return Clip255(a + ((b - c) >> 2));
}

static WEBP_INLINE uint32_t ClampedAddSubtractFullWithCoefficient(
    uint32_t c0, uint32_t c1, uint32_t c2) {
  const int a = AddSubtractComponentFullWithCoefficient(
      c0 >> 24, c1 >> 24, c2 >> 24);
  const int r = AddSubtractComponentFullWithCoefficient((c0 >> 16) & 0xff,
                                                       (c1 >> 16) & 0xff,
                                                       (c2 >> 16) & 0xff);
  const int g = AddSubtractComponentFullWithCoefficient((c0 >> 8) & 0xff,
                                                       (c1 >> 8) & 0xff,
                                                       (c2 >> 8) & 0xff);
  const int b = AddSubtractComponentFullWithCoefficient(
      c0 & 0xff, c1 & 0xff, c2 & 0xff);
  return ((uint32_t)a << 24) | (r << 16) | (g << 8) | b;
}

//------------------------------------------------------------------------------

// Find palette entry with minimum error from difference of actual pixel value
// and predicted pixel value. Propagate error of pixel to its top and left pixel
// in src array. Write predicted_value + palette_entry to new_image. Return
// index of best palette entry.
static int FindBestPaletteEntry(uint32_t src, uint32_t predicted_value,
                                const uint32_t palette[], int palette_size) {
  int i;
  int idx = 0;
  uint32_t best_distance = CalcDist(predicted_value, src, palette[0]);
  for (i = 1; i < palette_size; ++i) {
    const uint32_t distance = CalcDist(predicted_value, src, palette[i]);
    if (distance < best_distance) {
      best_distance = distance;
      idx = i;
    }
  }
  return idx;
}

static void ApplyBestPaletteEntry(int x, int y,
                                  uint32_t new_value, uint32_t palette_value,
                                  uint32_t* src, int src_stride,
                                  uint32_t* new_image) {
  AddPixelsEq(&new_value, palette_value);
  if (x > 0) {
    src[x - 1] = ClampedAddSubtractFullWithCoefficient(src[x - 1],
                                                       new_value, src[x]);
  }
  if (y > 0) {
    src[x - src_stride] =
        ClampedAddSubtractFullWithCoefficient(src[x - src_stride],
                                              new_value, src[x]);
  }
  new_image[x] = new_value;
}

//------------------------------------------------------------------------------
// Main entry point

static WebPEncodingError ApplyDeltaPalette(uint32_t* src, uint32_t* dst,
                                           uint32_t src_stride,
                                           uint32_t dst_stride,
                                           const uint32_t* palette,
                                           int palette_size,
                                           int width, int height,
                                           int num_passes) {
  int x, y;
  WebPEncodingError err = VP8_ENC_OK;
  uint32_t* new_image = (uint32_t*)WebPSafeMalloc(width, sizeof(*new_image));
  uint8_t* const tmp_row = (uint8_t*)WebPSafeMalloc(width, sizeof(*tmp_row));
  if (new_image == NULL || tmp_row == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  while (num_passes--) {
    uint32_t* cur_src = src;
    uint32_t* cur_dst = dst;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const uint32_t predicted_value = Predict(x, y, new_image);
        tmp_row[x] = FindBestPaletteEntry(cur_src[x], predicted_value,
                                          palette, palette_size);
        ApplyBestPaletteEntry(x, y, predicted_value, palette[tmp_row[x]],
                              cur_src, src_stride, new_image);
      }
      for (x = 0; x < width; ++x) {
        cur_dst[x] = palette[tmp_row[x]];
      }
      cur_src += src_stride;
      cur_dst += dst_stride;
    }
  }
 Error:
  WebPSafeFree(new_image);
  WebPSafeFree(tmp_row);
  return err;
}

// replaces enc->argb_ by a palettizable approximation of it,
// and generates optimal enc->palette_[]
WebPEncodingError WebPSearchOptimalDeltaPalette(VP8LEncoder* const enc) {
  const WebPPicture* const pic = enc->pic_;
  uint32_t* src = pic->argb;
  uint32_t* dst = enc->argb_;
  const int width = pic->width;
  const int height = pic->height;

  WebPEncodingError err = VP8_ENC_OK;
  memcpy(enc->palette_, kDeltaPalette, sizeof(kDeltaPalette));
  enc->palette_[DELTA_PALETTE_SIZE - 1] = src[0] - 0xff000000u;
  enc->palette_size_ = DELTA_PALETTE_SIZE;
  err = ApplyDeltaPalette(src, dst, pic->argb_stride, enc->current_width_,
                          enc->palette_, enc->palette_size_,
                          width, height, 2);
  if (err != VP8_ENC_OK) goto Error;

 Error:
  return err;
}

#else  // !WEBP_EXPERIMENTAL_FEATURES

WebPEncodingError WebPSearchOptimalDeltaPalette(VP8LEncoder* const enc) {
  (void)enc;
  return VP8_ENC_ERROR_INVALID_CONFIGURATION;
}

#endif  // WEBP_EXPERIMENTAL_FEATURES
