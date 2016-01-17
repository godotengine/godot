// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
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
  const uint64_t total_size = (uint64_t)sizeof(*set)
                            + size * sizeof(*set->histograms)
                            + size * sizeof(**set->histograms);
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
    int code, extra_bits_count, extra_bits_value;
    PrefixEncode(PixOrCopyLength(v),
                 &code, &extra_bits_count, &extra_bits_value);
    ++histo->literal_[256 + code];
    PrefixEncode(PixOrCopyDistance(v),
                 &code, &extra_bits_count, &extra_bits_value);
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

double VP8LHistogramEstimateBitsBulk(const VP8LHistogram* const p) {
  double retval = BitsEntropy(&p->literal_[0], VP8LHistogramNumCodes(p))
                + BitsEntropy(&p->red_[0], 256)
                + BitsEntropy(&p->blue_[0], 256)
                + BitsEntropy(&p->alpha_[0], 256)
                + BitsEntropy(&p->distance_[0], NUM_DISTANCE_CODES);
  // Compute the extra bits cost.
  int i;
  for (i = 2; i < NUM_LENGTH_CODES - 2; ++i) {
    retval +=
        (i >> 1) * p->literal_[256 + i + 2];
  }
  for (i = 2; i < NUM_DISTANCE_CODES - 2; ++i) {
    retval += (i >> 1) * p->distance_[i + 2];
  }
  return retval;
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

// Estimates the Huffman dictionary + other block overhead size.
static double HistogramEstimateBitsHeader(const VP8LHistogram* const p) {
  return HuffmanCost(&p->alpha_[0], 256) +
         HuffmanCost(&p->red_[0], 256) +
         HuffmanCost(&p->literal_[0], VP8LHistogramNumCodes(p)) +
         HuffmanCost(&p->blue_[0], 256) +
         HuffmanCost(&p->distance_[0], NUM_DISTANCE_CODES);
}

double VP8LHistogramEstimateBits(const VP8LHistogram* const p) {
  return HistogramEstimateBitsHeader(p) + VP8LHistogramEstimateBitsBulk(p);
}

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
                            VP8LHistogramSet* const out, int num_pairs) {
  int ok = 0;
  int i, iter;
  uint32_t seed = 0;
  int tries_with_no_success = 0;
  const int min_cluster_size = 2;
  int out_size = in->size;
  const int outer_iters = in->size * 3;
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
    // We pick the best pair to be combined out of 'inner_iters' pairs.
    double best_cost_diff = 0.;
    int best_idx1 = 0, best_idx2 = 1;
    int j;
    seed += iter;
    for (j = 0; j < num_pairs; ++j) {
      double curr_cost_diff;
      // Choose two histograms at random and try to combine them.
      const uint32_t idx1 = MyRand(&seed) % out_size;
      const uint32_t tmp = ((j & 7) + 1) % (out_size - 1);
      const uint32_t diff = (tmp < 3) ? tmp : MyRand(&seed) % (out_size - 1);
      const uint32_t idx2 = (idx1 + diff + 1) % out_size;
      if (idx1 == idx2) {
        continue;
      }
      *cur_combo = *out->histograms[idx1];
      VP8LHistogramAdd(cur_combo, out->histograms[idx2]);
      cur_combo->bit_cost_ = VP8LHistogramEstimateBits(cur_combo);
      // Calculate cost reduction on combining.
      curr_cost_diff = cur_combo->bit_cost_
                     - out->histograms[idx1]->bit_cost_
                     - out->histograms[idx2]->bit_cost_;
      if (best_cost_diff > curr_cost_diff) {    // found a better pair?
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

    if (best_cost_diff < 0.0) {
      *out->histograms[best_idx1] = *best_combo;
      // swap best_idx2 slot with last one (which is now unused)
      --out_size;
      if (best_idx2 != out_size) {
        out->histograms[best_idx2] = out->histograms[out_size];
        out->histograms[out_size] = NULL;   // just for sanity check.
      }
      tries_with_no_success = 0;
    }
    if (++tries_with_no_success >= 50) {
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

// What is the bit cost of moving square_histogram from
// cur_symbol to candidate_symbol.
// TODO(skal): we don't really need to copy the histogram and Add(). Instead
// we just need VP8LDualHistogramEstimateBits(A, B) estimation function.
static double HistogramDistance(const VP8LHistogram* const square_histogram,
                                const VP8LHistogram* const candidate) {
  const double previous_bit_cost = candidate->bit_cost_;
  double new_bit_cost;
  VP8LHistogram modified_histo;
  modified_histo = *candidate;
  VP8LHistogramAdd(&modified_histo, square_histogram);
  new_bit_cost = VP8LHistogramEstimateBits(&modified_histo);

  return new_bit_cost - previous_bit_cost;
}

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
static void HistogramRemap(const VP8LHistogramSet* const in,
                           const VP8LHistogramSet* const out,
                           uint16_t* const symbols) {
  int i;
  for (i = 0; i < in->size; ++i) {
    int best_out = 0;
    double best_bits = HistogramDistance(in->histograms[i], out->histograms[0]);
    int k;
    for (k = 1; k < out->size; ++k) {
      const double cur_bits =
          HistogramDistance(in->histograms[i], out->histograms[k]);
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
    VP8LHistogramAdd(out->histograms[symbols[i]], in->histograms[i]);
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
  const int num_histo_pairs = 10 + quality / 2;  // For HistogramCombine().
  const int histo_image_raw_size = histo_xsize * histo_ysize;
  VP8LHistogramSet* const image_out =
      VP8LAllocateHistogramSet(histo_image_raw_size, cache_bits);
  if (image_out == NULL) return 0;

  // Build histogram image.
  HistogramBuildImage(xsize, histo_bits, refs, image_out);
  // Collapse similar histograms.
  if (!HistogramCombine(image_out, image_in, num_histo_pairs)) {
    goto Error;
  }
  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(image_out, image_in, histogram_symbols);
  ok = 1;

Error:
  free(image_out);
  return ok;
}
