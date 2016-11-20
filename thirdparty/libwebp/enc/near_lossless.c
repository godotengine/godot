// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Near-lossless image preprocessing adjusts pixel values to help
// compressibility with a guarantee of maximum deviation between original and
// resulting pixel values.
//
// Author: Jyrki Alakuijala (jyrki@google.com)
// Converted to C by Aleksander Kramarz (akramarz@google.com)

#include <assert.h>
#include <stdlib.h>

#include "../dsp/lossless.h"
#include "../utils/utils.h"
#include "./vp8enci.h"

#define MIN_DIM_FOR_NEAR_LOSSLESS 64
#define MAX_LIMIT_BITS             5

// Quantizes the value up or down to a multiple of 1<<bits (or to 255),
// choosing the closer one, resolving ties using bankers' rounding.
static int FindClosestDiscretized(int a, int bits) {
  const int mask = (1 << bits) - 1;
  const int biased = a + (mask >> 1) + ((a >> bits) & 1);
  assert(bits > 0);
  if (biased > 0xff) return 0xff;
  return biased & ~mask;
}

// Applies FindClosestDiscretized to all channels of pixel.
static uint32_t ClosestDiscretizedArgb(uint32_t a, int bits) {
  return
      (FindClosestDiscretized(a >> 24, bits) << 24) |
      (FindClosestDiscretized((a >> 16) & 0xff, bits) << 16) |
      (FindClosestDiscretized((a >> 8) & 0xff, bits) << 8) |
      (FindClosestDiscretized(a & 0xff, bits));
}

// Checks if distance between corresponding channel values of pixels a and b
// is within the given limit.
static int IsNear(uint32_t a, uint32_t b, int limit) {
  int k;
  for (k = 0; k < 4; ++k) {
    const int delta =
        (int)((a >> (k * 8)) & 0xff) - (int)((b >> (k * 8)) & 0xff);
    if (delta >= limit || delta <= -limit) {
      return 0;
    }
  }
  return 1;
}

static int IsSmooth(const uint32_t* const prev_row,
                    const uint32_t* const curr_row,
                    const uint32_t* const next_row,
                    int ix, int limit) {
  // Check that all pixels in 4-connected neighborhood are smooth.
  return (IsNear(curr_row[ix], curr_row[ix - 1], limit) &&
          IsNear(curr_row[ix], curr_row[ix + 1], limit) &&
          IsNear(curr_row[ix], prev_row[ix], limit) &&
          IsNear(curr_row[ix], next_row[ix], limit));
}

// Adjusts pixel values of image with given maximum error.
static void NearLossless(int xsize, int ysize, uint32_t* argb,
                         int limit_bits, uint32_t* copy_buffer) {
  int x, y;
  const int limit = 1 << limit_bits;
  uint32_t* prev_row = copy_buffer;
  uint32_t* curr_row = prev_row + xsize;
  uint32_t* next_row = curr_row + xsize;
  memcpy(copy_buffer, argb, xsize * 2 * sizeof(argb[0]));

  for (y = 1; y < ysize - 1; ++y) {
    uint32_t* const curr_argb_row = argb + y * xsize;
    uint32_t* const next_argb_row = curr_argb_row + xsize;
    memcpy(next_row, next_argb_row, xsize * sizeof(argb[0]));
    for (x = 1; x < xsize - 1; ++x) {
      if (!IsSmooth(prev_row, curr_row, next_row, x, limit)) {
        curr_argb_row[x] = ClosestDiscretizedArgb(curr_row[x], limit_bits);
      }
    }
    {
      // Three-way swap.
      uint32_t* const temp = prev_row;
      prev_row = curr_row;
      curr_row = next_row;
      next_row = temp;
    }
  }
}

int VP8ApplyNearLossless(int xsize, int ysize, uint32_t* argb, int quality) {
  int i;
  uint32_t* const copy_buffer =
      (uint32_t*)WebPSafeMalloc(xsize * 3, sizeof(*copy_buffer));
  const int limit_bits = VP8LNearLosslessBits(quality);
  assert(argb != NULL);
  assert(limit_bits >= 0);
  assert(limit_bits <= MAX_LIMIT_BITS);
  if (copy_buffer == NULL) {
    return 0;
  }
  // For small icon images, don't attempt to apply near-lossless compression.
  if (xsize < MIN_DIM_FOR_NEAR_LOSSLESS && ysize < MIN_DIM_FOR_NEAR_LOSSLESS) {
    WebPSafeFree(copy_buffer);
    return 1;
  }

  for (i = limit_bits; i != 0; --i) {
    NearLossless(xsize, ysize, argb, i, copy_buffer);
  }
  WebPSafeFree(copy_buffer);
  return 1;
}
