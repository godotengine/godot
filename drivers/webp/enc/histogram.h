// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//
// Models the histograms of literal and distance codes.

#ifndef WEBP_ENC_HISTOGRAM_H_
#define WEBP_ENC_HISTOGRAM_H_

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "./backward_references.h"
#include "../webp/format_constants.h"
#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// A simple container for histograms of data.
typedef struct {
  // literal_ contains green literal, palette-code and
  // copy-length-prefix histogram
  int literal_[PIX_OR_COPY_CODES_MAX];
  int red_[256];
  int blue_[256];
  int alpha_[256];
  // Backward reference prefix-code histogram.
  int distance_[NUM_DISTANCE_CODES];
  int palette_code_bits_;
  double bit_cost_;   // cached value of VP8LHistogramEstimateBits(this)
} VP8LHistogram;

// Collection of histograms with fixed capacity, allocated as one
// big memory chunk. Can be destroyed by simply calling 'free()'.
typedef struct {
  int size;         // number of slots currently in use
  int max_size;     // maximum capacity
  VP8LHistogram** histograms;
} VP8LHistogramSet;

// Create the histogram.
//
// The input data is the PixOrCopy data, which models the literals, stop
// codes and backward references (both distances and lengths).  Also: if
// palette_code_bits is >= 0, initialize the histogram with this value.
void VP8LHistogramCreate(VP8LHistogram* const p,
                         const VP8LBackwardRefs* const refs,
                         int palette_code_bits);

// Set the palette_code_bits and reset the stats.
void VP8LHistogramInit(VP8LHistogram* const p, int palette_code_bits);

// Collect all the references into a histogram (without reset)
void VP8LHistogramStoreRefs(const VP8LBackwardRefs* const refs,
                            VP8LHistogram* const histo);

// Allocate an array of pointer to histograms, allocated and initialized
// using 'cache_bits'. Return NULL in case of memory error.
VP8LHistogramSet* VP8LAllocateHistogramSet(int size, int cache_bits);

// Accumulate a token 'v' into a histogram.
void VP8LHistogramAddSinglePixOrCopy(VP8LHistogram* const histo,
                                     const PixOrCopy* const v);

// Estimate how many bits the combined entropy of literals and distance
// approximately maps to.
double VP8LHistogramEstimateBits(const VP8LHistogram* const p);

// This function estimates the cost in bits excluding the bits needed to
// represent the entropy code itself.
double VP8LHistogramEstimateBitsBulk(const VP8LHistogram* const p);

static WEBP_INLINE void VP8LHistogramAdd(VP8LHistogram* const p,
                                         const VP8LHistogram* const a) {
  int i;
  for (i = 0; i < PIX_OR_COPY_CODES_MAX; ++i) {
    p->literal_[i] += a->literal_[i];
  }
  for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
    p->distance_[i] += a->distance_[i];
  }
  for (i = 0; i < 256; ++i) {
    p->red_[i] += a->red_[i];
    p->blue_[i] += a->blue_[i];
    p->alpha_[i] += a->alpha_[i];
  }
}

static WEBP_INLINE int VP8LHistogramNumCodes(const VP8LHistogram* const p) {
  return 256 + NUM_LENGTH_CODES +
      ((p->palette_code_bits_ > 0) ? (1 << p->palette_code_bits_) : 0);
}

// Builds the histogram image.
int VP8LGetHistoImageSymbols(int xsize, int ysize,
                             const VP8LBackwardRefs* const refs,
                             int quality, int histogram_bits, int cache_bits,
                             VP8LHistogramSet* const image_in,
                             uint16_t* const histogram_symbols);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif  // WEBP_ENC_HISTOGRAM_H_
