// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>

#include "./backward_references.h"
#include "./histogram.h"
#include "../dsp/lossless.h"
#include "../utils/utils.h"

static void HistogramClear(VP8LHistogram* const p) {
  memset(p->literal_, 0, sizeof(p->literal_));
  memset(p->red_, 0, sizeof(p->red_));
  memset(p->blue_, 0, sizeof(p->blue_));
  memset(p->alpha_, 0, sizeof(p->alpha_));
  memset(p->distance_, 0, sizeof(p->distance_));
  p->bit_cost_ = 0;
}

void VP8LHistogramStoreRefs(const VP8LBackwardRefs* const refs,
                            VP8LHistogram* const histo) {
  int i;
  for (i = 0; i < refs->size; ++i) {
    VP8LHistogramAddSinglePixOrCopy(histo, &refs->refs[i]);
  }
}

void VP8LHistogramCreate(VP8LHistogram* const p,
                         const VP8LBackwardRefs* const refs,
                         int palette_code_bits) {
  if (palette_code_bits >= 0) {
    p->palette_code_bits_ = palette_code_bits;
  }
  HistogramClear(p);
  VP8LHistogramStoreRefs(refs, p);
}

void VP8LHistogramInit(VP8LHistogram* const p, int palette_code_bits) {
  p->palette_code_bits_ = palette_code_bits;
  HistogramClear(p);
}

VP8LHistogramSet* VP8LAllocateHistogramSet(int size, int cache_bits) {
  int i;
  VP8LHistogramSet* set;
  VP8LHistogram* bulk;
  const uint64_t total_size = sizeof(*set)
                            + (uint64_t)size * sizeof(*set->histograms)
                            + (uint64_t)size * sizeof(**set->histograms);
  uint8_t* memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;

  set = (VP8LHistogramSet*)memory;
  memory += sizeof(*set);
  set->histograms = (VP8LHistogram**)memory;
  memory += size * sizeof(*set->histograms);
  bulk = (VP8LHistogram*)memory;
  set->max_size = size;
  set->size = size;
  for (i = 0; i < size; ++i) {
    set->histograms[i] = bulk + i;
    VP8LHistogramInit(set->histograms[i], cache_bits);
  }
  return set;
}

// -----------------------------------------------------------------------------

void VP8LHistogramAddSinglePixOrCopy(VP8LHistogram* const histo,
                                     const PixOrCopy* const v) {
  if (PixOrCopyIsLiteral(v)) {
    ++histo->alpha_[PixOrCopyLiteral(v, 3)];
    ++histo->red_[PixOrCopyLiteral(v, 2)];
    ++histo->literal_[PixOrCopyLiteral(v, 1)];
    ++histo->blue_[PixOrCopyLiteral(v, 0)];
  } else if (PixOrCopyIsCacheIdx(v)) {
    int literal_ix = 256 + NUM_LENGTH_CODES + PixOrCopyCacheIdx(v);
    ++histo->literal_[literal_ix];
  } else {
    int code, extra_bits;
    VP8LPrefixEncodeBits(PixOrCopyLength(v), &code, &extra_bits);
    ++histo->literal_[256 + code];
    VP8LPrefixEncodeBits(PixOrCopyDistance(v), &code, &extra_bits);
    ++histo->distance_[code];
  }
}

static double BitsEntropy(const int* const array, int n) {
  double retval = 0.;
  int sum = 0;
  int nonzeros = 0;
  int max_val = 0;
  int i;
  double mix;
  for (i = 0; i < n; ++i) {
    if (array[i] != 0) {
      sum += array[i];
      ++nonzeros;
      retval -= VP8LFastSLog2(array[i]);
      if (max_val < array[i]) {
        max_val = array[i];
      }
    }
  }
  retval += VP8LFastSLog2(sum);

  if (nonzeros < 5) {
    if (nonzeros <= 1) {
      return 0;
    }
    // Two symbols, they will be 0 and 1 in a Huffman code.
    // Let's mix in a bit of entropy to favor good clustering when
    // distributions of these are combined.
    if (nonzeros == 2) {
      return 0.99 * sum + 0.01 * retval;
    }
    // No matter what the entropy says, we cannot be better than min_limit
    // with Huffman coding. I am mixing a bit of entropy into the
    // min_limit since it produces much better (~0.5 %) compression results
    // perhaps because of better entropy clustering.
    if (nonzeros == 3) {
      mix = 0.95;
    } else {
      mix = 0.7;  // nonzeros == 4.
    }
  } else {
    mix = 0.627;
  }

  {
    double min_limit = 2 * sum - max_val;
    min_limit = mix * min_limit + (1.0 - mix) * retval;
    return (retval < min_limit) ? min_limit : retval;
  }
}

// Returns the cost encode the rle-encoded entropy code.
// The constants in this function are experimental.
static double HuffmanCost(const int* const population, int length) {
  // Small bias because Huffman code length is typically not stored in
  // full length.
  static const int kHuffmanCodeOfHuffmanCodeSize = CODE_LENGTH_CODES * 3;
  static const double kSmallBias = 9.1;
  double retval = kHuffmanCodeOfHuffmanCodeSize - kSmallBias;
  int streak = 0;
  int i = 0;
  for (; i < length - 1; ++i) {
    ++streak;
    if (population[i] == population[i + 1]) {
      continue;
    }
 last_streak_hack:
    // population[i] points now to the symbol in the streak of same values.
    if (streak > 3) {
      if (population[i] == 0) {
        retval += 1.5625 + 0.234375 * streak;
      } else {
        retval += 2.578125 + 0.703125 * streak;
      }
    } else {
      if (population[i] == 0) {
        retval += 1.796875 * streak;
      } else {
        retval += 3.28125 * streak;
      }
    }
    streak = 0;
  }
  if (i == length - 1) {
    ++streak;
    goto last_streak_hack;
  }
  return retval;
}

static double PopulationCost(const int* const population, int length) {
  return BitsEntropy(population, length) + HuffmanCost(population, length);
}

static double ExtraCost(const int* const population, int length) {
  int i;
  double cost = 0.;
  for (i = 2; i < length - 2; ++i) cost += (i >> 1) * population[i + 2];
  return cost;
}

// Estimates the Entropy + Huffman + other block overhead size cost.
double VP8LHistogramEstimateBits(const VP8LHistogram* const p) {
  return PopulationCost(p->literal_, VP8LHistogramNumCodes(p))
       + PopulationCost(p->red_, 256)
       + PopulationCost(p->blue_, 256)
       + PopulationCost(p->alpha_, 256)
       + PopulationCost(p->distance_, NUM_DISTANCE_CODES)
       + ExtraCost(p->literal_ + 256, NUM_LENGTH_CODES)
       + ExtraCost(p->distance_, NUM_DISTANCE_CODES);
}

double VP8LHistogramEstimateBitsBulk(const VP8LHistogram* const p) {
  return BitsEntropy(p->literal_, VP8LHistogramNumCodes(p))
       + BitsEntropy(p->red_, 256)
       + BitsEntropy(p->blue_, 256)
       + BitsEntropy(p->alpha_, 256)
       + BitsEntropy(p->distance_, NUM_DISTANCE_CODES)
       + ExtraCost(p->literal_ + 256, NUM_LENGTH_CODES)
       + ExtraCost(p->distance_, NUM_DISTANCE_CODES);
}

// -----------------------------------------------------------------------------
// Various histogram combine/cost-eval functions

// Adds 'in' histogram to 'out'
static void HistogramAdd(const VP8LHistogram* const in,
                         VP8LHistogram* const out) {
  int i;
  for (i = 0; i < PIX_OR_COPY_CODES_MAX; ++i) {
    out->literal_[i] += in->literal_[i];
  }
  for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
    out->distance_[i] += in->distance_[i];
  }
  for (i = 0; i < 256; ++i) {
    out->red_[i] += in->red_[i];
    out->blue_[i] += in->blue_[i];
    out->alpha_[i] += in->alpha_[i];
  }
}

// Performs out = a + b, computing the cost C(a+b) - C(a) - C(b) while comparing
// to the threshold value 'cost_threshold'. The score returned is
//  Score = C(a+b) - C(a) - C(b), where C(a) + C(b) is known and fixed.
// Since the previous score passed is 'cost_threshold', we only need to compare
// the partial cost against 'cost_threshold + C(a) + C(b)' to possibly bail-out
// early.
static double HistogramAddEval(const VP8LHistogram* const a,
                               const VP8LHistogram* const b,
                               VP8LHistogram* const out,
                               double cost_threshold) {
  double cost = 0;
  const double sum_cost = a->bit_cost_ + b->bit_cost_;
  int i;

  cost_threshold += sum_cost;

  // palette_code_bits_ is part of the cost evaluation for literal_.
  // TODO(skal): remove/simplify this palette_code_bits_?
  out->palette_code_bits_ =
      (a->palette_code_bits_ > b->palette_code_bits_) ? a->palette_code_bits_ :
                                                        b->palette_code_bits_;
  for (i = 0; i < PIX_OR_COPY_CODES_MAX; ++i) {
    out->literal_[i] = a->literal_[i] + b->literal_[i];
  }
  cost += PopulationCost(out->literal_, VP8LHistogramNumCodes(out));
  cost += ExtraCost(out->literal_ + 256, NUM_LENGTH_CODES);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) out->red_[i] = a->red_[i] + b->red_[i];
  cost += PopulationCost(out->red_, 256);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) out->blue_[i] = a->blue_[i] + b->blue_[i];
  cost += PopulationCost(out->blue_, 256);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
    out->distance_[i] = a->distance_[i] + b->distance_[i];
  }
  cost += PopulationCost(out->distance_, NUM_DISTANCE_CODES);
  cost += ExtraCost(out->distance_, NUM_DISTANCE_CODES);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) out->alpha_[i] = a->alpha_[i] + b->alpha_[i];
  cost += PopulationCost(out->alpha_, 256);

  out->bit_cost_ = cost;
  return cost - sum_cost;
}

// Same as HistogramAddEval(), except that the resulting histogram
// is not stored. Only the cost C(a+b) - C(a) is evaluated. We omit
// the term C(b) which is constant over all the evaluations.
static double HistogramAddThresh(const VP8LHistogram* const a,
                                 const VP8LHistogram* const b,
                                 double cost_threshold) {
  int tmp[PIX_OR_COPY_CODES_MAX];  // <= max storage we'll need
  int i;
  double cost = -a->bit_cost_;

  for (i = 0; i < PIX_OR_COPY_CODES_MAX; ++i) {
    tmp[i] = a->literal_[i] + b->literal_[i];
  }
  // note that the tests are ordered so that the usually largest
  // cost shares come first.
  cost += PopulationCost(tmp, VP8LHistogramNumCodes(a));
  cost += ExtraCost(tmp + 256, NUM_LENGTH_CODES);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) tmp[i] = a->red_[i] + b->red_[i];
  cost += PopulationCost(tmp, 256);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) tmp[i] = a->blue_[i] + b->blue_[i];
  cost += PopulationCost(tmp, 256);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
    tmp[i] = a->distance_[i] + b->distance_[i];
  }
  cost += PopulationCost(tmp, NUM_DISTANCE_CODES);
  cost += ExtraCost(tmp, NUM_DISTANCE_CODES);
  if (cost > cost_threshold) return cost;

  for (i = 0; i < 256; ++i) tmp[i] = a->alpha_[i] + b->alpha_[i];
  cost += PopulationCost(tmp, 256);

  return cost;
}

// -----------------------------------------------------------------------------

static void HistogramBuildImage(int xsize, int histo_bits,
                                const VP8LBackwardRefs* const backward_refs,
                                VP8LHistogramSet* const image) {
  int i;
  int x = 0, y = 0;
  const int histo_xsize = VP8LSubSampleSize(xsize, histo_bits);
  VP8LHistogram** const histograms = image->histograms;
  assert(histo_bits > 0);
  for (i = 0; i < backward_refs->size; ++i) {
    const PixOrCopy* const v = &backward_refs->refs[i];
    const int ix = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
    VP8LHistogramAddSinglePixOrCopy(histograms[ix], v);
    x += PixOrCopyLength(v);
    while (x >= xsize) {
      x -= xsize;
      ++y;
    }
  }
}

static uint32_t MyRand(uint32_t *seed) {
  *seed *= 16807U;
  if (*seed == 0) {
    *seed = 1;
  }
  return *seed;
}

static int HistogramCombine(const VP8LHistogramSet* const in,
                            VP8LHistogramSet* const out, int iter_mult,
                            int num_pairs, int num_tries_no_success) {
  int ok = 0;
  int i, iter;
  uint32_t seed = 0;
  int tries_with_no_success = 0;
  int out_size = in->size;
  const int outer_iters = in->size * iter_mult;
  const int min_cluster_size = 2;
  VP8LHistogram* const histos = (VP8LHistogram*)malloc(2 * sizeof(*histos));
  VP8LHistogram* cur_combo = histos + 0;    // trial merged histogram
  VP8LHistogram* best_combo = histos + 1;   // best merged histogram so far
  if (histos == NULL) goto End;

  // Copy histograms from in[] to out[].
  assert(in->size <= out->size);
  for (i = 0; i < in->size; ++i) {
    in->histograms[i]->bit_cost_ = VP8LHistogramEstimateBits(in->histograms[i]);
    *out->histograms[i] = *in->histograms[i];
  }

  // Collapse similar histograms in 'out'.
  for (iter = 0; iter < outer_iters && out_size >= min_cluster_size; ++iter) {
    double best_cost_diff = 0.;
    int best_idx1 = -1, best_idx2 = 1;
    int j;
    const int num_tries = (num_pairs < out_size) ? num_pairs : out_size;
    seed += iter;
    for (j = 0; j < num_tries; ++j) {
      double curr_cost_diff;
      // Choose two histograms at random and try to combine them.
      const uint32_t idx1 = MyRand(&seed) % out_size;
      const uint32_t tmp = (j & 7) + 1;
      const uint32_t diff = (tmp < 3) ? tmp : MyRand(&seed) % (out_size - 1);
      const uint32_t idx2 = (idx1 + diff + 1) % out_size;
      if (idx1 == idx2) {
        continue;
      }
      // Calculate cost reduction on combining.
      curr_cost_diff = HistogramAddEval(out->histograms[idx1],
                                        out->histograms[idx2],
                                        cur_combo, best_cost_diff);
      if (curr_cost_diff < best_cost_diff) {    // found a better pair?
        {     // swap cur/best combo histograms
          VP8LHistogram* const tmp_histo = cur_combo;
          cur_combo = best_combo;
          best_combo = tmp_histo;
        }
        best_cost_diff = curr_cost_diff;
        best_idx1 = idx1;
        best_idx2 = idx2;
      }
    }

    if (best_idx1 >= 0) {
      *out->histograms[best_idx1] = *best_combo;
      // swap best_idx2 slot with last one (which is now unused)
      --out_size;
      if (best_idx2 != out_size) {
        out->histograms[best_idx2] = out->histograms[out_size];
        out->histograms[out_size] = NULL;   // just for sanity check.
      }
      tries_with_no_success = 0;
    }
    if (++tries_with_no_success >= num_tries_no_success) {
      break;
    }
  }
  out->size = out_size;
  ok = 1;

 End:
  free(histos);
  return ok;
}

// -----------------------------------------------------------------------------
// Histogram refinement

// What is the bit cost of moving square_histogram from cur_symbol to candidate.
static double HistogramDistance(const VP8LHistogram* const square_histogram,
                                const VP8LHistogram* const candidate,
                                double cost_threshold) {
  return HistogramAddThresh(candidate, square_histogram, cost_threshold);
}

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
static void HistogramRemap(const VP8LHistogramSet* const in,
                           const VP8LHistogramSet* const out,
                           uint16_t* const symbols) {
  int i;
  for (i = 0; i < in->size; ++i) {
    int best_out = 0;
    double best_bits =
        HistogramDistance(in->histograms[i], out->histograms[0], 1.e38);
    int k;
    for (k = 1; k < out->size; ++k) {
      const double cur_bits =
          HistogramDistance(in->histograms[i], out->histograms[k], best_bits);
      if (cur_bits < best_bits) {
        best_bits = cur_bits;
        best_out = k;
      }
    }
    symbols[i] = best_out;
  }

  // Recompute each out based on raw and symbols.
  for (i = 0; i < out->size; ++i) {
    HistogramClear(out->histograms[i]);
  }
  for (i = 0; i < in->size; ++i) {
    HistogramAdd(in->histograms[i], out->histograms[symbols[i]]);
  }
}

int VP8LGetHistoImageSymbols(int xsize, int ysize,
                             const VP8LBackwardRefs* const refs,
                             int quality, int histo_bits, int cache_bits,
                             VP8LHistogramSet* const image_in,
                             uint16_t* const histogram_symbols) {
  int ok = 0;
  const int histo_xsize = histo_bits ? VP8LSubSampleSize(xsize, histo_bits) : 1;
  const int histo_ysize = histo_bits ? VP8LSubSampleSize(ysize, histo_bits) : 1;
  const int histo_image_raw_size = histo_xsize * histo_ysize;

  // Heuristic params for HistogramCombine().
  const int num_tries_no_success = 8 + (quality >> 1);
  const int iter_mult = (quality < 27) ? 1 : 1 + ((quality - 27) >> 4);
  const int num_pairs = (quality < 25) ? 10 : (5 * quality) >> 3;

  VP8LHistogramSet* const image_out =
      VP8LAllocateHistogramSet(histo_image_raw_size, cache_bits);
  if (image_out == NULL) return 0;

  // Build histogram image.
  HistogramBuildImage(xsize, histo_bits, refs, image_out);
  // Collapse similar histograms.
  if (!HistogramCombine(image_out, image_in, iter_mult, num_pairs,
                        num_tries_no_success)) {
    goto Error;
  }
  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(image_out, image_in, histogram_symbols);
  ok = 1;

Error:
  free(image_out);
  return ok;
}
