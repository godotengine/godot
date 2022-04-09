// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transform methods for lossless encoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)
//          Vincent Rabaud (vrabaud@google.com)

#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/vp8li_enc.h"

#define MAX_DIFF_COST (1e30f)

static const float kSpatialPredictorBias = 15.f;
static const int kPredLowEffort = 11;
static const uint32_t kMaskAlpha = 0xff000000;

// Mostly used to reduce code size + readability
static WEBP_INLINE int GetMin(int a, int b) { return (a > b) ? b : a; }

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

static float PredictionCostSpatial(const int counts[256], int weight_0,
                                   double exp_val) {
  const int significant_symbols = 256 >> 4;
  const double exp_decay_factor = 0.6;
  double bits = weight_0 * counts[0];
  int i;
  for (i = 1; i < significant_symbols; ++i) {
    bits += exp_val * (counts[i] + counts[256 - i]);
    exp_val *= exp_decay_factor;
  }
  return (float)(-0.1 * bits);
}

static float PredictionCostSpatialHistogram(const int accumulated[4][256],
                                            const int tile[4][256]) {
  int i;
  double retval = 0;
  for (i = 0; i < 4; ++i) {
    const double kExpValue = 0.94;
    retval += PredictionCostSpatial(tile[i], 1, kExpValue);
    retval += VP8LCombinedShannonEntropy(tile[i], accumulated[i]);
  }
  return (float)retval;
}

static WEBP_INLINE void UpdateHisto(int histo_argb[4][256], uint32_t argb) {
  ++histo_argb[0][argb >> 24];
  ++histo_argb[1][(argb >> 16) & 0xff];
  ++histo_argb[2][(argb >> 8) & 0xff];
  ++histo_argb[3][argb & 0xff];
}

//------------------------------------------------------------------------------
// Spatial transform functions.

static WEBP_INLINE void PredictBatch(int mode, int x_start, int y,
                                     int num_pixels, const uint32_t* current,
                                     const uint32_t* upper, uint32_t* out) {
  if (x_start == 0) {
    if (y == 0) {
      // ARGB_BLACK.
      VP8LPredictorsSub[0](current, NULL, 1, out);
    } else {
      // Top one.
      VP8LPredictorsSub[2](current, upper, 1, out);
    }
    ++x_start;
    ++out;
    --num_pixels;
  }
  if (y == 0) {
    // Left one.
    VP8LPredictorsSub[1](current + x_start, NULL, num_pixels, out);
  } else {
    VP8LPredictorsSub[mode](current + x_start, upper + x_start, num_pixels,
                            out);
  }
}

#if (WEBP_NEAR_LOSSLESS == 1)
static WEBP_INLINE int GetMax(int a, int b) { return (a < b) ? b : a; }

static int MaxDiffBetweenPixels(uint32_t p1, uint32_t p2) {
  const int diff_a = abs((int)(p1 >> 24) - (int)(p2 >> 24));
  const int diff_r = abs((int)((p1 >> 16) & 0xff) - (int)((p2 >> 16) & 0xff));
  const int diff_g = abs((int)((p1 >> 8) & 0xff) - (int)((p2 >> 8) & 0xff));
  const int diff_b = abs((int)(p1 & 0xff) - (int)(p2 & 0xff));
  return GetMax(GetMax(diff_a, diff_r), GetMax(diff_g, diff_b));
}

static int MaxDiffAroundPixel(uint32_t current, uint32_t up, uint32_t down,
                              uint32_t left, uint32_t right) {
  const int diff_up = MaxDiffBetweenPixels(current, up);
  const int diff_down = MaxDiffBetweenPixels(current, down);
  const int diff_left = MaxDiffBetweenPixels(current, left);
  const int diff_right = MaxDiffBetweenPixels(current, right);
  return GetMax(GetMax(diff_up, diff_down), GetMax(diff_left, diff_right));
}

static uint32_t AddGreenToBlueAndRed(uint32_t argb) {
  const uint32_t green = (argb >> 8) & 0xff;
  uint32_t red_blue = argb & 0x00ff00ffu;
  red_blue += (green << 16) | green;
  red_blue &= 0x00ff00ffu;
  return (argb & 0xff00ff00u) | red_blue;
}

static void MaxDiffsForRow(int width, int stride, const uint32_t* const argb,
                           uint8_t* const max_diffs, int used_subtract_green) {
  uint32_t current, up, down, left, right;
  int x;
  if (width <= 2) return;
  current = argb[0];
  right = argb[1];
  if (used_subtract_green) {
    current = AddGreenToBlueAndRed(current);
    right = AddGreenToBlueAndRed(right);
  }
  // max_diffs[0] and max_diffs[width - 1] are never used.
  for (x = 1; x < width - 1; ++x) {
    up = argb[-stride + x];
    down = argb[stride + x];
    left = current;
    current = right;
    right = argb[x + 1];
    if (used_subtract_green) {
      up = AddGreenToBlueAndRed(up);
      down = AddGreenToBlueAndRed(down);
      right = AddGreenToBlueAndRed(right);
    }
    max_diffs[x] = MaxDiffAroundPixel(current, up, down, left, right);
  }
}

// Quantize the difference between the actual component value and its prediction
// to a multiple of quantization, working modulo 256, taking care not to cross
// a boundary (inclusive upper limit).
static uint8_t NearLosslessComponent(uint8_t value, uint8_t predict,
                                     uint8_t boundary, int quantization) {
  const int residual = (value - predict) & 0xff;
  const int boundary_residual = (boundary - predict) & 0xff;
  const int lower = residual & ~(quantization - 1);
  const int upper = lower + quantization;
  // Resolve ties towards a value closer to the prediction (i.e. towards lower
  // if value comes after prediction and towards upper otherwise).
  const int bias = ((boundary - value) & 0xff) < boundary_residual;
  if (residual - lower < upper - residual + bias) {
    // lower is closer to residual than upper.
    if (residual > boundary_residual && lower <= boundary_residual) {
      // Halve quantization step to avoid crossing boundary. This midpoint is
      // on the same side of boundary as residual because midpoint >= residual
      // (since lower is closer than upper) and residual is above the boundary.
      return lower + (quantization >> 1);
    }
    return lower;
  } else {
    // upper is closer to residual than lower.
    if (residual <= boundary_residual && upper > boundary_residual) {
      // Halve quantization step to avoid crossing boundary. This midpoint is
      // on the same side of boundary as residual because midpoint <= residual
      // (since upper is closer than lower) and residual is below the boundary.
      return lower + (quantization >> 1);
    }
    return upper & 0xff;
  }
}

static WEBP_INLINE uint8_t NearLosslessDiff(uint8_t a, uint8_t b) {
  return (uint8_t)((((int)(a) - (int)(b))) & 0xff);
}

// Quantize every component of the difference between the actual pixel value and
// its prediction to a multiple of a quantization (a power of 2, not larger than
// max_quantization which is a power of 2, smaller than max_diff). Take care if
// value and predict have undergone subtract green, which means that red and
// blue are represented as offsets from green.
static uint32_t NearLossless(uint32_t value, uint32_t predict,
                             int max_quantization, int max_diff,
                             int used_subtract_green) {
  int quantization;
  uint8_t new_green = 0;
  uint8_t green_diff = 0;
  uint8_t a, r, g, b;
  if (max_diff <= 2) {
    return VP8LSubPixels(value, predict);
  }
  quantization = max_quantization;
  while (quantization >= max_diff) {
    quantization >>= 1;
  }
  if ((value >> 24) == 0 || (value >> 24) == 0xff) {
    // Preserve transparency of fully transparent or fully opaque pixels.
    a = NearLosslessDiff((value >> 24) & 0xff, (predict >> 24) & 0xff);
  } else {
    a = NearLosslessComponent(value >> 24, predict >> 24, 0xff, quantization);
  }
  g = NearLosslessComponent((value >> 8) & 0xff, (predict >> 8) & 0xff, 0xff,
                            quantization);
  if (used_subtract_green) {
    // The green offset will be added to red and blue components during decoding
    // to obtain the actual red and blue values.
    new_green = ((predict >> 8) + g) & 0xff;
    // The amount by which green has been adjusted during quantization. It is
    // subtracted from red and blue for compensation, to avoid accumulating two
    // quantization errors in them.
    green_diff = NearLosslessDiff(new_green, (value >> 8) & 0xff);
  }
  r = NearLosslessComponent(NearLosslessDiff((value >> 16) & 0xff, green_diff),
                            (predict >> 16) & 0xff, 0xff - new_green,
                            quantization);
  b = NearLosslessComponent(NearLosslessDiff(value & 0xff, green_diff),
                            predict & 0xff, 0xff - new_green, quantization);
  return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}
#endif  // (WEBP_NEAR_LOSSLESS == 1)

// Stores the difference between the pixel and its prediction in "out".
// In case of a lossy encoding, updates the source image to avoid propagating
// the deviation further to pixels which depend on the current pixel for their
// predictions.
static WEBP_INLINE void GetResidual(
    int width, int height, uint32_t* const upper_row,
    uint32_t* const current_row, const uint8_t* const max_diffs, int mode,
    int x_start, int x_end, int y, int max_quantization, int exact,
    int used_subtract_green, uint32_t* const out) {
  if (exact) {
    PredictBatch(mode, x_start, y, x_end - x_start, current_row, upper_row,
                 out);
  } else {
    const VP8LPredictorFunc pred_func = VP8LPredictors[mode];
    int x;
    for (x = x_start; x < x_end; ++x) {
      uint32_t predict;
      uint32_t residual;
      if (y == 0) {
        predict = (x == 0) ? ARGB_BLACK : current_row[x - 1];  // Left.
      } else if (x == 0) {
        predict = upper_row[x];  // Top.
      } else {
        predict = pred_func(&current_row[x - 1], upper_row + x);
      }
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization == 1 || mode == 0 || y == 0 || y == height - 1 ||
          x == 0 || x == width - 1) {
        residual = VP8LSubPixels(current_row[x], predict);
      } else {
        residual = NearLossless(current_row[x], predict, max_quantization,
                                max_diffs[x], used_subtract_green);
        // Update the source image.
        current_row[x] = VP8LAddPixels(predict, residual);
        // x is never 0 here so we do not need to update upper_row like below.
      }
#else
      (void)max_diffs;
      (void)height;
      (void)max_quantization;
      (void)used_subtract_green;
      residual = VP8LSubPixels(current_row[x], predict);
#endif
      if ((current_row[x] & kMaskAlpha) == 0) {
        // If alpha is 0, cleanup RGB. We can choose the RGB values of the
        // residual for best compression. The prediction of alpha itself can be
        // non-zero and must be kept though. We choose RGB of the residual to be
        // 0.
        residual &= kMaskAlpha;
        // Update the source image.
        current_row[x] = predict & ~kMaskAlpha;
        // The prediction for the rightmost pixel in a row uses the leftmost
        // pixel
        // in that row as its top-right context pixel. Hence if we change the
        // leftmost pixel of current_row, the corresponding change must be
        // applied
        // to upper_row as well where top-right context is being read from.
        if (x == 0 && y != 0) upper_row[width] = current_row[0];
      }
      out[x - x_start] = residual;
    }
  }
}

// Returns best predictor and updates the accumulated histogram.
// If max_quantization > 1, assumes that near lossless processing will be
// applied, quantizing residuals to multiples of quantization levels up to
// max_quantization (the actual quantization level depends on smoothness near
// the given pixel).
static int GetBestPredictorForTile(int width, int height,
                                   int tile_x, int tile_y, int bits,
                                   int accumulated[4][256],
                                   uint32_t* const argb_scratch,
                                   const uint32_t* const argb,
                                   int max_quantization,
                                   int exact, int used_subtract_green,
                                   const uint32_t* const modes) {
  const int kNumPredModes = 14;
  const int start_x = tile_x << bits;
  const int start_y = tile_y << bits;
  const int tile_size = 1 << bits;
  const int max_y = GetMin(tile_size, height - start_y);
  const int max_x = GetMin(tile_size, width - start_x);
  // Whether there exist columns just outside the tile.
  const int have_left = (start_x > 0);
  // Position and size of the strip covering the tile and adjacent columns if
  // they exist.
  const int context_start_x = start_x - have_left;
#if (WEBP_NEAR_LOSSLESS == 1)
  const int context_width = max_x + have_left + (max_x < width - start_x);
#endif
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  // Prediction modes of the left and above neighbor tiles.
  const int left_mode = (tile_x > 0) ?
      (modes[tile_y * tiles_per_row + tile_x - 1] >> 8) & 0xff : 0xff;
  const int above_mode = (tile_y > 0) ?
      (modes[(tile_y - 1) * tiles_per_row + tile_x] >> 8) & 0xff : 0xff;
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  uint32_t* upper_row = argb_scratch;
  uint32_t* current_row = upper_row + width + 1;
  uint8_t* const max_diffs = (uint8_t*)(current_row + width + 1);
  float best_diff = MAX_DIFF_COST;
  int best_mode = 0;
  int mode;
  int histo_stack_1[4][256];
  int histo_stack_2[4][256];
  // Need pointers to be able to swap arrays.
  int (*histo_argb)[256] = histo_stack_1;
  int (*best_histo)[256] = histo_stack_2;
  int i, j;
  uint32_t residuals[1 << MAX_TRANSFORM_BITS];
  assert(bits <= MAX_TRANSFORM_BITS);
  assert(max_x <= (1 << MAX_TRANSFORM_BITS));

  for (mode = 0; mode < kNumPredModes; ++mode) {
    float cur_diff;
    int relative_y;
    memset(histo_argb, 0, sizeof(histo_stack_1));
    if (start_y > 0) {
      // Read the row above the tile which will become the first upper_row.
      // Include a pixel to the left if it exists; include a pixel to the right
      // in all cases (wrapping to the leftmost pixel of the next row if it does
      // not exist).
      memcpy(current_row + context_start_x,
             argb + (start_y - 1) * width + context_start_x,
             sizeof(*argb) * (max_x + have_left + 1));
    }
    for (relative_y = 0; relative_y < max_y; ++relative_y) {
      const int y = start_y + relative_y;
      int relative_x;
      uint32_t* tmp = upper_row;
      upper_row = current_row;
      current_row = tmp;
      // Read current_row. Include a pixel to the left if it exists; include a
      // pixel to the right in all cases except at the bottom right corner of
      // the image (wrapping to the leftmost pixel of the next row if it does
      // not exist in the current row).
      memcpy(current_row + context_start_x,
             argb + y * width + context_start_x,
             sizeof(*argb) * (max_x + have_left + (y + 1 < height)));
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization > 1 && y >= 1 && y + 1 < height) {
        MaxDiffsForRow(context_width, width, argb + y * width + context_start_x,
                       max_diffs + context_start_x, used_subtract_green);
      }
#endif

      GetResidual(width, height, upper_row, current_row, max_diffs, mode,
                  start_x, start_x + max_x, y, max_quantization, exact,
                  used_subtract_green, residuals);
      for (relative_x = 0; relative_x < max_x; ++relative_x) {
        UpdateHisto(histo_argb, residuals[relative_x]);
      }
    }
    cur_diff = PredictionCostSpatialHistogram(
        (const int (*)[256])accumulated, (const int (*)[256])histo_argb);
    // Favor keeping the areas locally similar.
    if (mode == left_mode) cur_diff -= kSpatialPredictorBias;
    if (mode == above_mode) cur_diff -= kSpatialPredictorBias;

    if (cur_diff < best_diff) {
      int (*tmp)[256] = histo_argb;
      histo_argb = best_histo;
      best_histo = tmp;
      best_diff = cur_diff;
      best_mode = mode;
    }
  }

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 256; j++) {
      accumulated[i][j] += best_histo[i][j];
    }
  }

  return best_mode;
}

// Converts pixels of the image to residuals with respect to predictions.
// If max_quantization > 1, applies near lossless processing, quantizing
// residuals to multiples of quantization levels up to max_quantization
// (the actual quantization level depends on smoothness near the given pixel).
static void CopyImageWithPrediction(int width, int height,
                                    int bits, uint32_t* const modes,
                                    uint32_t* const argb_scratch,
                                    uint32_t* const argb,
                                    int low_effort, int max_quantization,
                                    int exact, int used_subtract_green) {
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  uint32_t* upper_row = argb_scratch;
  uint32_t* current_row = upper_row + width + 1;
  uint8_t* current_max_diffs = (uint8_t*)(current_row + width + 1);
#if (WEBP_NEAR_LOSSLESS == 1)
  uint8_t* lower_max_diffs = current_max_diffs + width;
#endif
  int y;

  for (y = 0; y < height; ++y) {
    int x;
    uint32_t* const tmp32 = upper_row;
    upper_row = current_row;
    current_row = tmp32;
    memcpy(current_row, argb + y * width,
           sizeof(*argb) * (width + (y + 1 < height)));

    if (low_effort) {
      PredictBatch(kPredLowEffort, 0, y, width, current_row, upper_row,
                   argb + y * width);
    } else {
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization > 1) {
        // Compute max_diffs for the lower row now, because that needs the
        // contents of argb for the current row, which we will overwrite with
        // residuals before proceeding with the next row.
        uint8_t* const tmp8 = current_max_diffs;
        current_max_diffs = lower_max_diffs;
        lower_max_diffs = tmp8;
        if (y + 2 < height) {
          MaxDiffsForRow(width, width, argb + (y + 1) * width, lower_max_diffs,
                         used_subtract_green);
        }
      }
#endif
      for (x = 0; x < width;) {
        const int mode =
            (modes[(y >> bits) * tiles_per_row + (x >> bits)] >> 8) & 0xff;
        int x_end = x + (1 << bits);
        if (x_end > width) x_end = width;
        GetResidual(width, height, upper_row, current_row, current_max_diffs,
                    mode, x, x_end, y, max_quantization, exact,
                    used_subtract_green, argb + y * width + x);
        x = x_end;
      }
    }
  }
}

// Finds the best predictor for each tile, and converts the image to residuals
// with respect to predictions. If near_lossless_quality < 100, applies
// near lossless processing, shaving off more bits of residuals for lower
// qualities.
void VP8LResidualImage(int width, int height, int bits, int low_effort,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image, int near_lossless_quality,
                       int exact, int used_subtract_green) {
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  const int tiles_per_col = VP8LSubSampleSize(height, bits);
  int tile_y;
  int histo[4][256];
  const int max_quantization = 1 << VP8LNearLosslessBits(near_lossless_quality);
  if (low_effort) {
    int i;
    for (i = 0; i < tiles_per_row * tiles_per_col; ++i) {
      image[i] = ARGB_BLACK | (kPredLowEffort << 8);
    }
  } else {
    memset(histo, 0, sizeof(histo));
    for (tile_y = 0; tile_y < tiles_per_col; ++tile_y) {
      int tile_x;
      for (tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
        const int pred = GetBestPredictorForTile(width, height, tile_x, tile_y,
            bits, histo, argb_scratch, argb, max_quantization, exact,
            used_subtract_green, image);
        image[tile_y * tiles_per_row + tile_x] = ARGB_BLACK | (pred << 8);
      }
    }
  }

  CopyImageWithPrediction(width, height, bits, image, argb_scratch, argb,
                          low_effort, max_quantization, exact,
                          used_subtract_green);
}

//------------------------------------------------------------------------------
// Color transform functions.

static WEBP_INLINE void MultipliersClear(VP8LMultipliers* const m) {
  m->green_to_red_ = 0;
  m->green_to_blue_ = 0;
  m->red_to_blue_ = 0;
}

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

static WEBP_INLINE uint32_t MultipliersToColorCode(
    const VP8LMultipliers* const m) {
  return 0xff000000u |
         ((uint32_t)(m->red_to_blue_) << 16) |
         ((uint32_t)(m->green_to_blue_) << 8) |
         m->green_to_red_;
}

static float PredictionCostCrossColor(const int accumulated[256],
                                      const int counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const double kExpValue = 2.4;
  return VP8LCombinedShannonEntropy(counts, accumulated) +
         PredictionCostSpatial(counts, 3, kExpValue);
}

static float GetPredictionCostCrossColorRed(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_red,
    const int accumulated_red_histo[256]) {
  int histo[256] = { 0 };
  float cur_diff;

  VP8LCollectColorRedTransforms(argb, stride, tile_width, tile_height,
                                green_to_red, histo);

  cur_diff = PredictionCostCrossColor(accumulated_red_histo, histo);
  if ((uint8_t)green_to_red == prev_x.green_to_red_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)green_to_red == prev_y.green_to_red_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_red == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

static void GetBestGreenToRed(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
    const int accumulated_red_histo[256], VP8LMultipliers* const best_tx) {
  const int kMaxIters = 4 + ((7 * quality) >> 8);  // in range [4..6]
  int green_to_red_best = 0;
  int iter, offset;
  float best_diff = GetPredictionCostCrossColorRed(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_red_best, accumulated_red_histo);
  for (iter = 0; iter < kMaxIters; ++iter) {
    // ColorTransformDelta is a 3.5 bit fixed point, so 32 is equal to
    // one in color computation. Having initial delta here as 1 is sufficient
    // to explore the range of (-2, 2).
    const int delta = 32 >> iter;
    // Try a negative and a positive delta from the best known value.
    for (offset = -delta; offset <= delta; offset += 2 * delta) {
      const int green_to_red_cur = offset + green_to_red_best;
      const float cur_diff = GetPredictionCostCrossColorRed(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_red_cur, accumulated_red_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_red_best = green_to_red_cur;
      }
    }
  }
  best_tx->green_to_red_ = (green_to_red_best & 0xff);
}

static float GetPredictionCostCrossColorBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y,
    int green_to_blue, int red_to_blue, const int accumulated_blue_histo[256]) {
  int histo[256] = { 0 };
  float cur_diff;

  VP8LCollectColorBlueTransforms(argb, stride, tile_width, tile_height,
                                 green_to_blue, red_to_blue, histo);

  cur_diff = PredictionCostCrossColor(accumulated_blue_histo, histo);
  if ((uint8_t)green_to_blue == prev_x.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)green_to_blue == prev_y.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_x.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_y.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_blue == 0) {
    cur_diff -= 3;
  }
  if (red_to_blue == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

#define kGreenRedToBlueNumAxis 8
#define kGreenRedToBlueMaxIters 7
static void GetBestGreenRedToBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
    const int accumulated_blue_histo[256],
    VP8LMultipliers* const best_tx) {
  const int8_t offset[kGreenRedToBlueNumAxis][2] =
      {{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int8_t delta_lut[kGreenRedToBlueMaxIters] = { 16, 16, 8, 4, 2, 2, 2 };
  const int iters =
      (quality < 25) ? 1 : (quality > 50) ? kGreenRedToBlueMaxIters : 4;
  int green_to_blue_best = 0;
  int red_to_blue_best = 0;
  int iter;
  // Initial value at origin:
  float best_diff = GetPredictionCostCrossColorBlue(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_blue_best, red_to_blue_best, accumulated_blue_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    int axis;
    for (axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          offset[axis][0] * delta + green_to_blue_best;
      const int red_to_blue_cur = offset[axis][1] * delta + red_to_blue_best;
      const float cur_diff = GetPredictionCostCrossColorBlue(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_blue_cur, red_to_blue_cur, accumulated_blue_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_blue_best = green_to_blue_cur;
        red_to_blue_best = red_to_blue_cur;
      }
      if (quality < 25 && iter == 4) {
        // Only axis aligned diffs for lower quality.
        break;  // next iter.
      }
    }
    if (delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0) {
      // Further iterations would not help.
      break;  // out of iter-loop.
    }
  }
  best_tx->green_to_blue_ = green_to_blue_best & 0xff;
  best_tx->red_to_blue_ = red_to_blue_best & 0xff;
}
#undef kGreenRedToBlueMaxIters
#undef kGreenRedToBlueNumAxis

static VP8LMultipliers GetBestColorTransformForTile(
    int tile_x, int tile_y, int bits,
    VP8LMultipliers prev_x,
    VP8LMultipliers prev_y,
    int quality, int xsize, int ysize,
    const int accumulated_red_histo[256],
    const int accumulated_blue_histo[256],
    const uint32_t* const argb) {
  const int max_tile_size = 1 << bits;
  const int tile_y_offset = tile_y * max_tile_size;
  const int tile_x_offset = tile_x * max_tile_size;
  const int all_x_max = GetMin(tile_x_offset + max_tile_size, xsize);
  const int all_y_max = GetMin(tile_y_offset + max_tile_size, ysize);
  const int tile_width = all_x_max - tile_x_offset;
  const int tile_height = all_y_max - tile_y_offset;
  const uint32_t* const tile_argb = argb + tile_y_offset * xsize
                                  + tile_x_offset;
  VP8LMultipliers best_tx;
  MultipliersClear(&best_tx);

  GetBestGreenToRed(tile_argb, xsize, tile_width, tile_height,
                    prev_x, prev_y, quality, accumulated_red_histo, &best_tx);
  GetBestGreenRedToBlue(tile_argb, xsize, tile_width, tile_height,
                        prev_x, prev_y, quality, accumulated_blue_histo,
                        &best_tx);
  return best_tx;
}

static void CopyTileWithColorTransform(int xsize, int ysize,
                                       int tile_x, int tile_y,
                                       int max_tile_size,
                                       VP8LMultipliers color_transform,
                                       uint32_t* argb) {
  const int xscan = GetMin(max_tile_size, xsize - tile_x);
  int yscan = GetMin(max_tile_size, ysize - tile_y);
  argb += tile_y * xsize + tile_x;
  while (yscan-- > 0) {
    VP8LTransformColor(&color_transform, argb, xscan);
    argb += xsize;
  }
}

void VP8LColorSpaceTransform(int width, int height, int bits, int quality,
                             uint32_t* const argb, uint32_t* image) {
  const int max_tile_size = 1 << bits;
  const int tile_xsize = VP8LSubSampleSize(width, bits);
  const int tile_ysize = VP8LSubSampleSize(height, bits);
  int accumulated_red_histo[256] = { 0 };
  int accumulated_blue_histo[256] = { 0 };
  int tile_x, tile_y;
  VP8LMultipliers prev_x, prev_y;
  MultipliersClear(&prev_y);
  MultipliersClear(&prev_x);
  for (tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      int y;
      const int tile_x_offset = tile_x * max_tile_size;
      const int tile_y_offset = tile_y * max_tile_size;
      const int all_x_max = GetMin(tile_x_offset + max_tile_size, width);
      const int all_y_max = GetMin(tile_y_offset + max_tile_size, height);
      const int offset = tile_y * tile_xsize + tile_x;
      if (tile_y != 0) {
        ColorCodeToMultipliers(image[offset - tile_xsize], &prev_y);
      }
      prev_x = GetBestColorTransformForTile(tile_x, tile_y, bits,
                                            prev_x, prev_y,
                                            quality, width, height,
                                            accumulated_red_histo,
                                            accumulated_blue_histo,
                                            argb);
      image[offset] = MultipliersToColorCode(&prev_x);
      CopyTileWithColorTransform(width, height, tile_x_offset, tile_y_offset,
                                 max_tile_size, prev_x, argb);

      // Gather accumulated histogram data.
      for (y = tile_y_offset; y < all_y_max; ++y) {
        int ix = y * width + tile_x_offset;
        const int ix_end = ix + all_x_max - tile_x_offset;
        for (; ix < ix_end; ++ix) {
          const uint32_t pix = argb[ix];
          if (ix >= 2 &&
              pix == argb[ix - 2] &&
              pix == argb[ix - 1]) {
            continue;  // repeated pixels are handled by backward references
          }
          if (ix >= width + 2 &&
              argb[ix - 2] == argb[ix - width - 2] &&
              argb[ix - 1] == argb[ix - width - 1] &&
              pix == argb[ix - width]) {
            continue;  // repeated pixels are handled by backward references
          }
          ++accumulated_red_histo[(pix >> 16) & 0xff];
          ++accumulated_blue_histo[(pix >> 0) & 0xff];
        }
      }
    }
  }
}
