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
#include "src/webp/config.h"
#endif

#include <math.h>

#include "src/enc/backward_references_enc.h"
#include "src/enc/histogram_enc.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/utils/utils.h"

#define MAX_COST 1.e38

// Number of partitions for the three dominant (literal, red and blue) symbol
// costs.
#define NUM_PARTITIONS 4
// The size of the bin-hash corresponding to the three dominant costs.
#define BIN_SIZE (NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS)
// Maximum number of histograms allowed in greedy combining algorithm.
#define MAX_HISTO_GREEDY 100

static void HistogramClear(VP8LHistogram* const p) {
  uint32_t* const literal = p->literal_;
  const int cache_bits = p->palette_code_bits_;
  const int histo_size = VP8LGetHistogramSize(cache_bits);
  memset(p, 0, histo_size);
  p->palette_code_bits_ = cache_bits;
  p->literal_ = literal;
}

// Swap two histogram pointers.
static void HistogramSwap(VP8LHistogram** const A, VP8LHistogram** const B) {
  VP8LHistogram* const tmp = *A;
  *A = *B;
  *B = tmp;
}

static void HistogramCopy(const VP8LHistogram* const src,
                          VP8LHistogram* const dst) {
  uint32_t* const dst_literal = dst->literal_;
  const int dst_cache_bits = dst->palette_code_bits_;
  const int literal_size = VP8LHistogramNumCodes(dst_cache_bits);
  const int histo_size = VP8LGetHistogramSize(dst_cache_bits);
  assert(src->palette_code_bits_ == dst_cache_bits);
  memcpy(dst, src, histo_size);
  dst->literal_ = dst_literal;
  memcpy(dst->literal_, src->literal_, literal_size * sizeof(*dst->literal_));
}

int VP8LGetHistogramSize(int cache_bits) {
  const int literal_size = VP8LHistogramNumCodes(cache_bits);
  const size_t total_size = sizeof(VP8LHistogram) + sizeof(int) * literal_size;
  assert(total_size <= (size_t)0x7fffffff);
  return (int)total_size;
}

void VP8LFreeHistogram(VP8LHistogram* const histo) {
  WebPSafeFree(histo);
}

void VP8LFreeHistogramSet(VP8LHistogramSet* const histo) {
  WebPSafeFree(histo);
}

void VP8LHistogramStoreRefs(const VP8LBackwardRefs* const refs,
                            VP8LHistogram* const histo) {
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    VP8LHistogramAddSinglePixOrCopy(histo, c.cur_pos, NULL, 0);
    VP8LRefsCursorNext(&c);
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

void VP8LHistogramInit(VP8LHistogram* const p, int palette_code_bits,
                       int init_arrays) {
  p->palette_code_bits_ = palette_code_bits;
  if (init_arrays) {
    HistogramClear(p);
  } else {
    p->trivial_symbol_ = 0;
    p->bit_cost_ = 0.;
    p->literal_cost_ = 0.;
    p->red_cost_ = 0.;
    p->blue_cost_ = 0.;
    memset(p->is_used_, 0, sizeof(p->is_used_));
  }
}

VP8LHistogram* VP8LAllocateHistogram(int cache_bits) {
  VP8LHistogram* histo = NULL;
  const int total_size = VP8LGetHistogramSize(cache_bits);
  uint8_t* const memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;
  histo = (VP8LHistogram*)memory;
  // literal_ won't necessary be aligned.
  histo->literal_ = (uint32_t*)(memory + sizeof(VP8LHistogram));
  VP8LHistogramInit(histo, cache_bits, /*init_arrays=*/ 0);
  return histo;
}

// Resets the pointers of the histograms to point to the bit buffer in the set.
static void HistogramSetResetPointers(VP8LHistogramSet* const set,
                                      int cache_bits) {
  int i;
  const int histo_size = VP8LGetHistogramSize(cache_bits);
  uint8_t* memory = (uint8_t*) (set->histograms);
  memory += set->max_size * sizeof(*set->histograms);
  for (i = 0; i < set->max_size; ++i) {
    memory = (uint8_t*) WEBP_ALIGN(memory);
    set->histograms[i] = (VP8LHistogram*) memory;
    // literal_ won't necessary be aligned.
    set->histograms[i]->literal_ = (uint32_t*)(memory + sizeof(VP8LHistogram));
    memory += histo_size;
  }
}

// Returns the total size of the VP8LHistogramSet.
static size_t HistogramSetTotalSize(int size, int cache_bits) {
  const int histo_size = VP8LGetHistogramSize(cache_bits);
  return (sizeof(VP8LHistogramSet) + size * (sizeof(VP8LHistogram*) +
          histo_size + WEBP_ALIGN_CST));
}

VP8LHistogramSet* VP8LAllocateHistogramSet(int size, int cache_bits) {
  int i;
  VP8LHistogramSet* set;
  const size_t total_size = HistogramSetTotalSize(size, cache_bits);
  uint8_t* memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;

  set = (VP8LHistogramSet*)memory;
  memory += sizeof(*set);
  set->histograms = (VP8LHistogram**)memory;
  set->max_size = size;
  set->size = size;
  HistogramSetResetPointers(set, cache_bits);
  for (i = 0; i < size; ++i) {
    VP8LHistogramInit(set->histograms[i], cache_bits, /*init_arrays=*/ 0);
  }
  return set;
}

void VP8LHistogramSetClear(VP8LHistogramSet* const set) {
  int i;
  const int cache_bits = set->histograms[0]->palette_code_bits_;
  const int size = set->size;
  const size_t total_size = HistogramSetTotalSize(size, cache_bits);
  uint8_t* memory = (uint8_t*)set;

  memset(memory, 0, total_size);
  memory += sizeof(*set);
  set->histograms = (VP8LHistogram**)memory;
  set->max_size = size;
  set->size = size;
  HistogramSetResetPointers(set, cache_bits);
  for (i = 0; i < size; ++i) {
    set->histograms[i]->palette_code_bits_ = cache_bits;
  }
}

// -----------------------------------------------------------------------------

void VP8LHistogramAddSinglePixOrCopy(VP8LHistogram* const histo,
                                     const PixOrCopy* const v,
                                     int (*const distance_modifier)(int, int),
                                     int distance_modifier_arg0) {
  if (PixOrCopyIsLiteral(v)) {
    ++histo->alpha_[PixOrCopyLiteral(v, 3)];
    ++histo->red_[PixOrCopyLiteral(v, 2)];
    ++histo->literal_[PixOrCopyLiteral(v, 1)];
    ++histo->blue_[PixOrCopyLiteral(v, 0)];
  } else if (PixOrCopyIsCacheIdx(v)) {
    const int literal_ix =
        NUM_LITERAL_CODES + NUM_LENGTH_CODES + PixOrCopyCacheIdx(v);
    ++histo->literal_[literal_ix];
  } else {
    int code, extra_bits;
    VP8LPrefixEncodeBits(PixOrCopyLength(v), &code, &extra_bits);
    ++histo->literal_[NUM_LITERAL_CODES + code];
    if (distance_modifier == NULL) {
      VP8LPrefixEncodeBits(PixOrCopyDistance(v), &code, &extra_bits);
    } else {
      VP8LPrefixEncodeBits(
          distance_modifier(distance_modifier_arg0, PixOrCopyDistance(v)),
          &code, &extra_bits);
    }
    ++histo->distance_[code];
  }
}

// -----------------------------------------------------------------------------
// Entropy-related functions.

static WEBP_INLINE double BitsEntropyRefine(const VP8LBitEntropy* entropy) {
  double mix;
  if (entropy->nonzeros < 5) {
    if (entropy->nonzeros <= 1) {
      return 0;
    }
    // Two symbols, they will be 0 and 1 in a Huffman code.
    // Let's mix in a bit of entropy to favor good clustering when
    // distributions of these are combined.
    if (entropy->nonzeros == 2) {
      return 0.99 * entropy->sum + 0.01 * entropy->entropy;
    }
    // No matter what the entropy says, we cannot be better than min_limit
    // with Huffman coding. I am mixing a bit of entropy into the
    // min_limit since it produces much better (~0.5 %) compression results
    // perhaps because of better entropy clustering.
    if (entropy->nonzeros == 3) {
      mix = 0.95;
    } else {
      mix = 0.7;  // nonzeros == 4.
    }
  } else {
    mix = 0.627;
  }

  {
    double min_limit = 2 * entropy->sum - entropy->max_val;
    min_limit = mix * min_limit + (1.0 - mix) * entropy->entropy;
    return (entropy->entropy < min_limit) ? min_limit : entropy->entropy;
  }
}

double VP8LBitsEntropy(const uint32_t* const array, int n) {
  VP8LBitEntropy entropy;
  VP8LBitsEntropyUnrefined(array, n, &entropy);

  return BitsEntropyRefine(&entropy);
}

static double InitialHuffmanCost(void) {
  // Small bias because Huffman code length is typically not stored in
  // full length.
  static const int kHuffmanCodeOfHuffmanCodeSize = CODE_LENGTH_CODES * 3;
  static const double kSmallBias = 9.1;
  return kHuffmanCodeOfHuffmanCodeSize - kSmallBias;
}

// Finalize the Huffman cost based on streak numbers and length type (<3 or >=3)
static double FinalHuffmanCost(const VP8LStreaks* const stats) {
  // The constants in this function are experimental and got rounded from
  // their original values in 1/8 when switched to 1/1024.
  double retval = InitialHuffmanCost();
  // Second coefficient: Many zeros in the histogram are covered efficiently
  // by a run-length encode. Originally 2/8.
  retval += stats->counts[0] * 1.5625 + 0.234375 * stats->streaks[0][1];
  // Second coefficient: Constant values are encoded less efficiently, but still
  // RLE'ed. Originally 6/8.
  retval += stats->counts[1] * 2.578125 + 0.703125 * stats->streaks[1][1];
  // 0s are usually encoded more efficiently than non-0s.
  // Originally 15/8.
  retval += 1.796875 * stats->streaks[0][0];
  // Originally 26/8.
  retval += 3.28125 * stats->streaks[1][0];
  return retval;
}

// Get the symbol entropy for the distribution 'population'.
// Set 'trivial_sym', if there's only one symbol present in the distribution.
static double PopulationCost(const uint32_t* const population, int length,
                             uint32_t* const trivial_sym,
                             uint8_t* const is_used) {
  VP8LBitEntropy bit_entropy;
  VP8LStreaks stats;
  VP8LGetEntropyUnrefined(population, length, &bit_entropy, &stats);
  if (trivial_sym != NULL) {
    *trivial_sym = (bit_entropy.nonzeros == 1) ? bit_entropy.nonzero_code
                                               : VP8L_NON_TRIVIAL_SYM;
  }
  // The histogram is used if there is at least one non-zero streak.
  *is_used = (stats.streaks[1][0] != 0 || stats.streaks[1][1] != 0);

  return BitsEntropyRefine(&bit_entropy) + FinalHuffmanCost(&stats);
}

// trivial_at_end is 1 if the two histograms only have one element that is
// non-zero: both the zero-th one, or both the last one.
static WEBP_INLINE double GetCombinedEntropy(const uint32_t* const X,
                                             const uint32_t* const Y,
                                             int length, int is_X_used,
                                             int is_Y_used,
                                             int trivial_at_end) {
  VP8LStreaks stats;
  if (trivial_at_end) {
    // This configuration is due to palettization that transforms an indexed
    // pixel into 0xff000000 | (pixel << 8) in VP8LBundleColorMap.
    // BitsEntropyRefine is 0 for histograms with only one non-zero value.
    // Only FinalHuffmanCost needs to be evaluated.
    memset(&stats, 0, sizeof(stats));
    // Deal with the non-zero value at index 0 or length-1.
    stats.streaks[1][0] = 1;
    // Deal with the following/previous zero streak.
    stats.counts[0] = 1;
    stats.streaks[0][1] = length - 1;
    return FinalHuffmanCost(&stats);
  } else {
    VP8LBitEntropy bit_entropy;
    if (is_X_used) {
      if (is_Y_used) {
        VP8LGetCombinedEntropyUnrefined(X, Y, length, &bit_entropy, &stats);
      } else {
        VP8LGetEntropyUnrefined(X, length, &bit_entropy, &stats);
      }
    } else {
      if (is_Y_used) {
        VP8LGetEntropyUnrefined(Y, length, &bit_entropy, &stats);
      } else {
        memset(&stats, 0, sizeof(stats));
        stats.counts[0] = 1;
        stats.streaks[0][length > 3] = length;
        VP8LBitEntropyInit(&bit_entropy);
      }
    }

    return BitsEntropyRefine(&bit_entropy) + FinalHuffmanCost(&stats);
  }
}

// Estimates the Entropy + Huffman + other block overhead size cost.
double VP8LHistogramEstimateBits(VP8LHistogram* const p) {
  return
      PopulationCost(p->literal_, VP8LHistogramNumCodes(p->palette_code_bits_),
                     NULL, &p->is_used_[0])
      + PopulationCost(p->red_, NUM_LITERAL_CODES, NULL, &p->is_used_[1])
      + PopulationCost(p->blue_, NUM_LITERAL_CODES, NULL, &p->is_used_[2])
      + PopulationCost(p->alpha_, NUM_LITERAL_CODES, NULL, &p->is_used_[3])
      + PopulationCost(p->distance_, NUM_DISTANCE_CODES, NULL, &p->is_used_[4])
      + VP8LExtraCost(p->literal_ + NUM_LITERAL_CODES, NUM_LENGTH_CODES)
      + VP8LExtraCost(p->distance_, NUM_DISTANCE_CODES);
}

// -----------------------------------------------------------------------------
// Various histogram combine/cost-eval functions

static int GetCombinedHistogramEntropy(const VP8LHistogram* const a,
                                       const VP8LHistogram* const b,
                                       double cost_threshold,
                                       double* cost) {
  const int palette_code_bits = a->palette_code_bits_;
  int trivial_at_end = 0;
  assert(a->palette_code_bits_ == b->palette_code_bits_);
  *cost += GetCombinedEntropy(a->literal_, b->literal_,
                              VP8LHistogramNumCodes(palette_code_bits),
                              a->is_used_[0], b->is_used_[0], 0);
  *cost += VP8LExtraCostCombined(a->literal_ + NUM_LITERAL_CODES,
                                 b->literal_ + NUM_LITERAL_CODES,
                                 NUM_LENGTH_CODES);
  if (*cost > cost_threshold) return 0;

  if (a->trivial_symbol_ != VP8L_NON_TRIVIAL_SYM &&
      a->trivial_symbol_ == b->trivial_symbol_) {
    // A, R and B are all 0 or 0xff.
    const uint32_t color_a = (a->trivial_symbol_ >> 24) & 0xff;
    const uint32_t color_r = (a->trivial_symbol_ >> 16) & 0xff;
    const uint32_t color_b = (a->trivial_symbol_ >> 0) & 0xff;
    if ((color_a == 0 || color_a == 0xff) &&
        (color_r == 0 || color_r == 0xff) &&
        (color_b == 0 || color_b == 0xff)) {
      trivial_at_end = 1;
    }
  }

  *cost +=
      GetCombinedEntropy(a->red_, b->red_, NUM_LITERAL_CODES, a->is_used_[1],
                         b->is_used_[1], trivial_at_end);
  if (*cost > cost_threshold) return 0;

  *cost +=
      GetCombinedEntropy(a->blue_, b->blue_, NUM_LITERAL_CODES, a->is_used_[2],
                         b->is_used_[2], trivial_at_end);
  if (*cost > cost_threshold) return 0;

  *cost +=
      GetCombinedEntropy(a->alpha_, b->alpha_, NUM_LITERAL_CODES,
                         a->is_used_[3], b->is_used_[3], trivial_at_end);
  if (*cost > cost_threshold) return 0;

  *cost +=
      GetCombinedEntropy(a->distance_, b->distance_, NUM_DISTANCE_CODES,
                         a->is_used_[4], b->is_used_[4], 0);
  *cost +=
      VP8LExtraCostCombined(a->distance_, b->distance_, NUM_DISTANCE_CODES);
  if (*cost > cost_threshold) return 0;

  return 1;
}

static WEBP_INLINE void HistogramAdd(const VP8LHistogram* const a,
                                     const VP8LHistogram* const b,
                                     VP8LHistogram* const out) {
  VP8LHistogramAdd(a, b, out);
  out->trivial_symbol_ = (a->trivial_symbol_ == b->trivial_symbol_)
                       ? a->trivial_symbol_
                       : VP8L_NON_TRIVIAL_SYM;
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
  cost_threshold += sum_cost;

  if (GetCombinedHistogramEntropy(a, b, cost_threshold, &cost)) {
    HistogramAdd(a, b, out);
    out->bit_cost_ = cost;
    out->palette_code_bits_ = a->palette_code_bits_;
  }

  return cost - sum_cost;
}

// Same as HistogramAddEval(), except that the resulting histogram
// is not stored. Only the cost C(a+b) - C(a) is evaluated. We omit
// the term C(b) which is constant over all the evaluations.
static double HistogramAddThresh(const VP8LHistogram* const a,
                                 const VP8LHistogram* const b,
                                 double cost_threshold) {
  double cost = -a->bit_cost_;
  GetCombinedHistogramEntropy(a, b, cost_threshold, &cost);
  return cost;
}

// -----------------------------------------------------------------------------

// The structure to keep track of cost range for the three dominant entropy
// symbols.
// TODO(skal): Evaluate if float can be used here instead of double for
// representing the entropy costs.
typedef struct {
  double literal_max_;
  double literal_min_;
  double red_max_;
  double red_min_;
  double blue_max_;
  double blue_min_;
} DominantCostRange;

static void DominantCostRangeInit(DominantCostRange* const c) {
  c->literal_max_ = 0.;
  c->literal_min_ = MAX_COST;
  c->red_max_ = 0.;
  c->red_min_ = MAX_COST;
  c->blue_max_ = 0.;
  c->blue_min_ = MAX_COST;
}

static void UpdateDominantCostRange(
    const VP8LHistogram* const h, DominantCostRange* const c) {
  if (c->literal_max_ < h->literal_cost_) c->literal_max_ = h->literal_cost_;
  if (c->literal_min_ > h->literal_cost_) c->literal_min_ = h->literal_cost_;
  if (c->red_max_ < h->red_cost_) c->red_max_ = h->red_cost_;
  if (c->red_min_ > h->red_cost_) c->red_min_ = h->red_cost_;
  if (c->blue_max_ < h->blue_cost_) c->blue_max_ = h->blue_cost_;
  if (c->blue_min_ > h->blue_cost_) c->blue_min_ = h->blue_cost_;
}

static void UpdateHistogramCost(VP8LHistogram* const h) {
  uint32_t alpha_sym, red_sym, blue_sym;
  const double alpha_cost =
      PopulationCost(h->alpha_, NUM_LITERAL_CODES, &alpha_sym,
                     &h->is_used_[3]);
  const double distance_cost =
      PopulationCost(h->distance_, NUM_DISTANCE_CODES, NULL, &h->is_used_[4]) +
      VP8LExtraCost(h->distance_, NUM_DISTANCE_CODES);
  const int num_codes = VP8LHistogramNumCodes(h->palette_code_bits_);
  h->literal_cost_ =
      PopulationCost(h->literal_, num_codes, NULL, &h->is_used_[0]) +
          VP8LExtraCost(h->literal_ + NUM_LITERAL_CODES, NUM_LENGTH_CODES);
  h->red_cost_ =
      PopulationCost(h->red_, NUM_LITERAL_CODES, &red_sym, &h->is_used_[1]);
  h->blue_cost_ =
      PopulationCost(h->blue_, NUM_LITERAL_CODES, &blue_sym, &h->is_used_[2]);
  h->bit_cost_ = h->literal_cost_ + h->red_cost_ + h->blue_cost_ +
                 alpha_cost + distance_cost;
  if ((alpha_sym | red_sym | blue_sym) == VP8L_NON_TRIVIAL_SYM) {
    h->trivial_symbol_ = VP8L_NON_TRIVIAL_SYM;
  } else {
    h->trivial_symbol_ =
        ((uint32_t)alpha_sym << 24) | (red_sym << 16) | (blue_sym << 0);
  }
}

static int GetBinIdForEntropy(double min, double max, double val) {
  const double range = max - min;
  if (range > 0.) {
    const double delta = val - min;
    return (int)((NUM_PARTITIONS - 1e-6) * delta / range);
  } else {
    return 0;
  }
}

static int GetHistoBinIndex(const VP8LHistogram* const h,
                            const DominantCostRange* const c, int low_effort) {
  int bin_id = GetBinIdForEntropy(c->literal_min_, c->literal_max_,
                                  h->literal_cost_);
  assert(bin_id < NUM_PARTITIONS);
  if (!low_effort) {
    bin_id = bin_id * NUM_PARTITIONS
           + GetBinIdForEntropy(c->red_min_, c->red_max_, h->red_cost_);
    bin_id = bin_id * NUM_PARTITIONS
           + GetBinIdForEntropy(c->blue_min_, c->blue_max_, h->blue_cost_);
    assert(bin_id < BIN_SIZE);
  }
  return bin_id;
}

// Construct the histograms from backward references.
static void HistogramBuild(
    int xsize, int histo_bits, const VP8LBackwardRefs* const backward_refs,
    VP8LHistogramSet* const image_histo) {
  int x = 0, y = 0;
  const int histo_xsize = VP8LSubSampleSize(xsize, histo_bits);
  VP8LHistogram** const histograms = image_histo->histograms;
  VP8LRefsCursor c = VP8LRefsCursorInit(backward_refs);
  assert(histo_bits > 0);
  VP8LHistogramSetClear(image_histo);
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    const int ix = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
    VP8LHistogramAddSinglePixOrCopy(histograms[ix], v, NULL, 0);
    x += PixOrCopyLength(v);
    while (x >= xsize) {
      x -= xsize;
      ++y;
    }
    VP8LRefsCursorNext(&c);
  }
}

// Copies the histograms and computes its bit_cost.
static void HistogramCopyAndAnalyze(
    VP8LHistogramSet* const orig_histo, VP8LHistogramSet* const image_histo) {
  int i;
  const int histo_size = orig_histo->size;
  VP8LHistogram** const orig_histograms = orig_histo->histograms;
  VP8LHistogram** const histograms = image_histo->histograms;
  image_histo->size = 0;
  for (i = 0; i < histo_size; ++i) {
    VP8LHistogram* const histo = orig_histograms[i];
    UpdateHistogramCost(histo);

    // Skip the histogram if it is completely empty, which can happen for tiles
    // with no information (when they are skipped because of LZ77).
    if (!histo->is_used_[0] && !histo->is_used_[1] && !histo->is_used_[2]
        && !histo->is_used_[3] && !histo->is_used_[4]) {
      continue;
    }
    // Copy histograms from orig_histo[] to image_histo[].
    HistogramCopy(histo, histograms[image_histo->size++]);
  }
}

// Partition histograms to different entropy bins for three dominant (literal,
// red and blue) symbol costs and compute the histogram aggregate bit_cost.
static void HistogramAnalyzeEntropyBin(VP8LHistogramSet* const image_histo,
                                       uint16_t* const bin_map,
                                       int low_effort) {
  int i;
  VP8LHistogram** const histograms = image_histo->histograms;
  const int histo_size = image_histo->size;
  DominantCostRange cost_range;
  DominantCostRangeInit(&cost_range);

  // Analyze the dominant (literal, red and blue) entropy costs.
  for (i = 0; i < histo_size; ++i) {
    UpdateDominantCostRange(histograms[i], &cost_range);
  }

  // bin-hash histograms on three of the dominant (literal, red and blue)
  // symbol costs and store the resulting bin_id for each histogram.
  for (i = 0; i < histo_size; ++i) {
    bin_map[i] = GetHistoBinIndex(histograms[i], &cost_range, low_effort);
  }
}

// Compact image_histo[] by merging some histograms with same bin_id together if
// it's advantageous.
static void HistogramCombineEntropyBin(VP8LHistogramSet* const image_histo,
                                       VP8LHistogram* cur_combo,
                                       const uint16_t* const bin_map,
                                       int bin_map_size, int num_bins,
                                       double combine_cost_factor,
                                       int low_effort) {
  VP8LHistogram** const histograms = image_histo->histograms;
  int idx;
  // Work in-place: processed histograms are put at the beginning of
  // image_histo[]. At the end, we just have to truncate the array.
  int size = 0;
  struct {
    int16_t first;    // position of the histogram that accumulates all
                      // histograms with the same bin_id
    uint16_t num_combine_failures;   // number of combine failures per bin_id
  } bin_info[BIN_SIZE];

  assert(num_bins <= BIN_SIZE);
  for (idx = 0; idx < num_bins; ++idx) {
    bin_info[idx].first = -1;
    bin_info[idx].num_combine_failures = 0;
  }

  for (idx = 0; idx < bin_map_size; ++idx) {
    const int bin_id = bin_map[idx];
    const int first = bin_info[bin_id].first;
    assert(size <= idx);
    if (first == -1) {
      // just move histogram #idx to its final position
      histograms[size] = histograms[idx];
      bin_info[bin_id].first = size++;
    } else if (low_effort) {
      HistogramAdd(histograms[idx], histograms[first], histograms[first]);
    } else {
      // try to merge #idx into #first (both share the same bin_id)
      const double bit_cost = histograms[idx]->bit_cost_;
      const double bit_cost_thresh = -bit_cost * combine_cost_factor;
      const double curr_cost_diff =
          HistogramAddEval(histograms[first], histograms[idx],
                           cur_combo, bit_cost_thresh);
      if (curr_cost_diff < bit_cost_thresh) {
        // Try to merge two histograms only if the combo is a trivial one or
        // the two candidate histograms are already non-trivial.
        // For some images, 'try_combine' turns out to be false for a lot of
        // histogram pairs. In that case, we fallback to combining
        // histograms as usual to avoid increasing the header size.
        const int try_combine =
            (cur_combo->trivial_symbol_ != VP8L_NON_TRIVIAL_SYM) ||
            ((histograms[idx]->trivial_symbol_ == VP8L_NON_TRIVIAL_SYM) &&
             (histograms[first]->trivial_symbol_ == VP8L_NON_TRIVIAL_SYM));
        const int max_combine_failures = 32;
        if (try_combine ||
            bin_info[bin_id].num_combine_failures >= max_combine_failures) {
          // move the (better) merged histogram to its final slot
          HistogramSwap(&cur_combo, &histograms[first]);
        } else {
          histograms[size++] = histograms[idx];
          ++bin_info[bin_id].num_combine_failures;
        }
      } else {
        histograms[size++] = histograms[idx];
      }
    }
  }
  image_histo->size = size;
  if (low_effort) {
    // for low_effort case, update the final cost when everything is merged
    for (idx = 0; idx < size; ++idx) {
      UpdateHistogramCost(histograms[idx]);
    }
  }
}

// Implement a Lehmer random number generator with a multiplicative constant of
// 48271 and a modulo constant of 2^31 - 1.
static uint32_t MyRand(uint32_t* const seed) {
  *seed = (uint32_t)(((uint64_t)(*seed) * 48271u) % 2147483647u);
  assert(*seed > 0);
  return *seed;
}

// -----------------------------------------------------------------------------
// Histogram pairs priority queue

// Pair of histograms. Negative idx1 value means that pair is out-of-date.
typedef struct {
  int idx1;
  int idx2;
  double cost_diff;
  double cost_combo;
} HistogramPair;

typedef struct {
  HistogramPair* queue;
  int size;
  int max_size;
} HistoQueue;

static int HistoQueueInit(HistoQueue* const histo_queue, const int max_index) {
  histo_queue->size = 0;
  // max_index^2 for the queue size is safe. If you look at
  // HistogramCombineGreedy, and imagine that UpdateQueueFront always pushes
  // data to the queue, you insert at most:
  // - max_index*(max_index-1)/2 (the first two for loops)
  // - max_index - 1 in the last for loop at the first iteration of the while
  //   loop, max_index - 2 at the second iteration ... therefore
  //   max_index*(max_index-1)/2 overall too
  histo_queue->max_size = max_index * max_index;
  // We allocate max_size + 1 because the last element at index "size" is
  // used as temporary data (and it could be up to max_size).
  histo_queue->queue = (HistogramPair*)WebPSafeMalloc(
      histo_queue->max_size + 1, sizeof(*histo_queue->queue));
  return histo_queue->queue != NULL;
}

static void HistoQueueClear(HistoQueue* const histo_queue) {
  assert(histo_queue != NULL);
  WebPSafeFree(histo_queue->queue);
  histo_queue->size = 0;
  histo_queue->max_size = 0;
}

// Pop a specific pair in the queue by replacing it with the last one
// and shrinking the queue.
static void HistoQueuePopPair(HistoQueue* const histo_queue,
                              HistogramPair* const pair) {
  assert(pair >= histo_queue->queue &&
         pair < (histo_queue->queue + histo_queue->size));
  assert(histo_queue->size > 0);
  *pair = histo_queue->queue[histo_queue->size - 1];
  --histo_queue->size;
}

// Check whether a pair in the queue should be updated as head or not.
static void HistoQueueUpdateHead(HistoQueue* const histo_queue,
                                 HistogramPair* const pair) {
  assert(pair->cost_diff < 0.);
  assert(pair >= histo_queue->queue &&
         pair < (histo_queue->queue + histo_queue->size));
  assert(histo_queue->size > 0);
  if (pair->cost_diff < histo_queue->queue[0].cost_diff) {
    // Replace the best pair.
    const HistogramPair tmp = histo_queue->queue[0];
    histo_queue->queue[0] = *pair;
    *pair = tmp;
  }
}

// Update the cost diff and combo of a pair of histograms. This needs to be
// called when the the histograms have been merged with a third one.
static void HistoQueueUpdatePair(const VP8LHistogram* const h1,
                                 const VP8LHistogram* const h2,
                                 double threshold,
                                 HistogramPair* const pair) {
  const double sum_cost = h1->bit_cost_ + h2->bit_cost_;
  pair->cost_combo = 0.;
  GetCombinedHistogramEntropy(h1, h2, sum_cost + threshold, &pair->cost_combo);
  pair->cost_diff = pair->cost_combo - sum_cost;
}

// Create a pair from indices "idx1" and "idx2" provided its cost
// is inferior to "threshold", a negative entropy.
// It returns the cost of the pair, or 0. if it superior to threshold.
static double HistoQueuePush(HistoQueue* const histo_queue,
                             VP8LHistogram** const histograms, int idx1,
                             int idx2, double threshold) {
  const VP8LHistogram* h1;
  const VP8LHistogram* h2;
  HistogramPair pair;

  assert(threshold <= 0.);
  if (idx1 > idx2) {
    const int tmp = idx2;
    idx2 = idx1;
    idx1 = tmp;
  }
  pair.idx1 = idx1;
  pair.idx2 = idx2;
  h1 = histograms[idx1];
  h2 = histograms[idx2];

  HistoQueueUpdatePair(h1, h2, threshold, &pair);

  // Do not even consider the pair if it does not improve the entropy.
  if (pair.cost_diff >= threshold) return 0.;

  // We cannot add more elements than the capacity.
  assert(histo_queue->size < histo_queue->max_size);
  histo_queue->queue[histo_queue->size++] = pair;
  HistoQueueUpdateHead(histo_queue, &histo_queue->queue[histo_queue->size - 1]);

  return pair.cost_diff;
}

// -----------------------------------------------------------------------------

// Combines histograms by continuously choosing the one with the highest cost
// reduction.
static int HistogramCombineGreedy(VP8LHistogramSet* const image_histo) {
  int ok = 0;
  int image_histo_size = image_histo->size;
  int i, j;
  VP8LHistogram** const histograms = image_histo->histograms;
  // Indexes of remaining histograms.
  int* const clusters =
      (int*)WebPSafeMalloc(image_histo_size, sizeof(*clusters));
  // Priority queue of histogram pairs.
  HistoQueue histo_queue;

  if (!HistoQueueInit(&histo_queue, image_histo_size) || clusters == NULL) {
    goto End;
  }

  for (i = 0; i < image_histo_size; ++i) {
    // Initialize clusters indexes.
    clusters[i] = i;
    for (j = i + 1; j < image_histo_size; ++j) {
      // Initialize positions array.
      HistoQueuePush(&histo_queue, histograms, i, j, 0.);
    }
  }

  while (image_histo_size > 1 && histo_queue.size > 0) {
    const int idx1 = histo_queue.queue[0].idx1;
    const int idx2 = histo_queue.queue[0].idx2;
    HistogramAdd(histograms[idx2], histograms[idx1], histograms[idx1]);
    histograms[idx1]->bit_cost_ = histo_queue.queue[0].cost_combo;
    // Remove merged histogram.
    for (i = 0; i + 1 < image_histo_size; ++i) {
      if (clusters[i] >= idx2) {
        clusters[i] = clusters[i + 1];
      }
    }
    --image_histo_size;

    // Remove pairs intersecting the just combined best pair.
    for (i = 0; i < histo_queue.size;) {
      HistogramPair* const p = histo_queue.queue + i;
      if (p->idx1 == idx1 || p->idx2 == idx1 ||
          p->idx1 == idx2 || p->idx2 == idx2) {
        HistoQueuePopPair(&histo_queue, p);
      } else {
        HistoQueueUpdateHead(&histo_queue, p);
        ++i;
      }
    }

    // Push new pairs formed with combined histogram to the queue.
    for (i = 0; i < image_histo_size; ++i) {
      if (clusters[i] != idx1) {
        HistoQueuePush(&histo_queue, histograms, idx1, clusters[i], 0.);
      }
    }
  }
  // Move remaining histograms to the beginning of the array.
  for (i = 0; i < image_histo_size; ++i) {
    if (i != clusters[i]) {  // swap the two histograms
      HistogramSwap(&histograms[i], &histograms[clusters[i]]);
    }
  }

  image_histo->size = image_histo_size;
  ok = 1;

 End:
  WebPSafeFree(clusters);
  HistoQueueClear(&histo_queue);
  return ok;
}

// Perform histogram aggregation using a stochastic approach.
// 'do_greedy' is set to 1 if a greedy approach needs to be performed
// afterwards, 0 otherwise.
static int HistogramCombineStochastic(VP8LHistogramSet* const image_histo,
                                      int min_cluster_size,
                                      int* const do_greedy) {
  int iter;
  uint32_t seed = 1;
  int tries_with_no_success = 0;
  int image_histo_size = image_histo->size;
  const int outer_iters = image_histo_size;
  const int num_tries_no_success = outer_iters / 2;
  VP8LHistogram** const histograms = image_histo->histograms;
  // Priority queue of histogram pairs. Its size of "kCostHeapSizeSqrt"^2
  // impacts the quality of the compression and the speed: the smaller the
  // faster but the worse for the compression.
  HistoQueue histo_queue;
  const int kHistoQueueSizeSqrt = 3;
  int ok = 0;

  if (!HistoQueueInit(&histo_queue, kHistoQueueSizeSqrt)) {
    goto End;
  }
  // Collapse similar histograms in 'image_histo'.
  ++min_cluster_size;
  for (iter = 0; iter < outer_iters && image_histo_size >= min_cluster_size &&
                 ++tries_with_no_success < num_tries_no_success;
       ++iter) {
    double best_cost =
        (histo_queue.size == 0) ? 0. : histo_queue.queue[0].cost_diff;
    int best_idx1 = -1, best_idx2 = 1;
    int j;
    const uint32_t rand_range = (image_histo_size - 1) * image_histo_size;
    // image_histo_size / 2 was chosen empirically. Less means faster but worse
    // compression.
    const int num_tries = image_histo_size / 2;

    for (j = 0; j < num_tries; ++j) {
      double curr_cost;
      // Choose two different histograms at random and try to combine them.
      const uint32_t tmp = MyRand(&seed) % rand_range;
      const uint32_t idx1 = tmp / (image_histo_size - 1);
      uint32_t idx2 = tmp % (image_histo_size - 1);
      if (idx2 >= idx1) ++idx2;

      // Calculate cost reduction on combination.
      curr_cost =
          HistoQueuePush(&histo_queue, histograms, idx1, idx2, best_cost);
      if (curr_cost < 0) {  // found a better pair?
        best_cost = curr_cost;
        // Empty the queue if we reached full capacity.
        if (histo_queue.size == histo_queue.max_size) break;
      }
    }
    if (histo_queue.size == 0) continue;

    // Merge the two best histograms.
    best_idx1 = histo_queue.queue[0].idx1;
    best_idx2 = histo_queue.queue[0].idx2;
    assert(best_idx1 < best_idx2);
    HistogramAddEval(histograms[best_idx1], histograms[best_idx2],
                     histograms[best_idx1], 0);
    // Swap the best_idx2 histogram with the last one (which is now unused).
    --image_histo_size;
    if (best_idx2 != image_histo_size) {
      HistogramSwap(&histograms[image_histo_size], &histograms[best_idx2]);
    }
    histograms[image_histo_size] = NULL;
    // Parse the queue and update each pair that deals with best_idx1,
    // best_idx2 or image_histo_size.
    for (j = 0; j < histo_queue.size;) {
      HistogramPair* const p = histo_queue.queue + j;
      const int is_idx1_best = p->idx1 == best_idx1 || p->idx1 == best_idx2;
      const int is_idx2_best = p->idx2 == best_idx1 || p->idx2 == best_idx2;
      int do_eval = 0;
      // The front pair could have been duplicated by a random pick so
      // check for it all the time nevertheless.
      if (is_idx1_best && is_idx2_best) {
        HistoQueuePopPair(&histo_queue, p);
        continue;
      }
      // Any pair containing one of the two best indices should only refer to
      // best_idx1. Its cost should also be updated.
      if (is_idx1_best) {
        p->idx1 = best_idx1;
        do_eval = 1;
      } else if (is_idx2_best) {
        p->idx2 = best_idx1;
        do_eval = 1;
      }
      if (p->idx2 == image_histo_size) {
        // No need to re-evaluate here as it does not involve a pair
        // containing best_idx1 or best_idx2.
        p->idx2 = best_idx2;
      }
      assert(p->idx2 < image_histo_size);
      // Make sure the index order is respected.
      if (p->idx1 > p->idx2) {
        const int tmp = p->idx2;
        p->idx2 = p->idx1;
        p->idx1 = tmp;
      }
      if (do_eval) {
        // Re-evaluate the cost of an updated pair.
        HistoQueueUpdatePair(histograms[p->idx1], histograms[p->idx2], 0., p);
        if (p->cost_diff >= 0.) {
          HistoQueuePopPair(&histo_queue, p);
          continue;
        }
      }
      HistoQueueUpdateHead(&histo_queue, p);
      ++j;
    }

    tries_with_no_success = 0;
  }
  image_histo->size = image_histo_size;
  *do_greedy = (image_histo->size <= min_cluster_size);
  ok = 1;

End:
  HistoQueueClear(&histo_queue);
  return ok;
}

// -----------------------------------------------------------------------------
// Histogram refinement

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
static void HistogramRemap(const VP8LHistogramSet* const in,
                           const VP8LHistogramSet* const out,
                           uint16_t* const symbols) {
  int i;
  VP8LHistogram** const in_histo = in->histograms;
  VP8LHistogram** const out_histo = out->histograms;
  const int in_size = in->size;
  const int out_size = out->size;
  if (out_size > 1) {
    for (i = 0; i < in_size; ++i) {
      int best_out = 0;
      double best_bits = MAX_COST;
      int k;
      for (k = 0; k < out_size; ++k) {
        const double cur_bits =
            HistogramAddThresh(out_histo[k], in_histo[i], best_bits);
        if (k == 0 || cur_bits < best_bits) {
          best_bits = cur_bits;
          best_out = k;
        }
      }
      symbols[i] = best_out;
    }
  } else {
    assert(out_size == 1);
    for (i = 0; i < in_size; ++i) {
      symbols[i] = 0;
    }
  }

  // Recompute each out based on raw and symbols.
  for (i = 0; i < out_size; ++i) {
    HistogramClear(out_histo[i]);
  }

  for (i = 0; i < in_size; ++i) {
    const int idx = symbols[i];
    HistogramAdd(in_histo[i], out_histo[idx], out_histo[idx]);
  }
}

static double GetCombineCostFactor(int histo_size, int quality) {
  double combine_cost_factor = 0.16;
  if (quality < 90) {
    if (histo_size > 256) combine_cost_factor /= 2.;
    if (histo_size > 512) combine_cost_factor /= 2.;
    if (histo_size > 1024) combine_cost_factor /= 2.;
    if (quality <= 50) combine_cost_factor /= 2.;
  }
  return combine_cost_factor;
}

int VP8LGetHistoImageSymbols(int xsize, int ysize,
                             const VP8LBackwardRefs* const refs,
                             int quality, int low_effort,
                             int histo_bits, int cache_bits,
                             VP8LHistogramSet* const image_histo,
                             VP8LHistogram* const tmp_histo,
                             uint16_t* const histogram_symbols) {
  int ok = 0;
  const int histo_xsize = histo_bits ? VP8LSubSampleSize(xsize, histo_bits) : 1;
  const int histo_ysize = histo_bits ? VP8LSubSampleSize(ysize, histo_bits) : 1;
  const int image_histo_raw_size = histo_xsize * histo_ysize;
  VP8LHistogramSet* const orig_histo =
      VP8LAllocateHistogramSet(image_histo_raw_size, cache_bits);
  // Don't attempt linear bin-partition heuristic for
  // histograms of small sizes (as bin_map will be very sparse) and
  // maximum quality q==100 (to preserve the compression gains at that level).
  const int entropy_combine_num_bins = low_effort ? NUM_PARTITIONS : BIN_SIZE;
  int entropy_combine;

  if (orig_histo == NULL) goto Error;

  // Construct the histograms from backward references.
  HistogramBuild(xsize, histo_bits, refs, orig_histo);
  // Copies the histograms and computes its bit_cost.
  HistogramCopyAndAnalyze(orig_histo, image_histo);
  entropy_combine =
      (image_histo->size > entropy_combine_num_bins * 2) && (quality < 100);
  if (entropy_combine) {
    const int bin_map_size = image_histo->size;
    // Reuse histogram_symbols storage. By definition, it's guaranteed to be ok.
    uint16_t* const bin_map = histogram_symbols;
    const double combine_cost_factor =
        GetCombineCostFactor(image_histo_raw_size, quality);

    HistogramAnalyzeEntropyBin(image_histo, bin_map, low_effort);
    // Collapse histograms with similar entropy.
    HistogramCombineEntropyBin(image_histo, tmp_histo, bin_map, bin_map_size,
                               entropy_combine_num_bins, combine_cost_factor,
                               low_effort);
  }

  // Don't combine the histograms using stochastic and greedy heuristics for
  // low-effort compression mode.
  if (!low_effort || !entropy_combine) {
    const float x = quality / 100.f;
    // cubic ramp between 1 and MAX_HISTO_GREEDY:
    const int threshold_size = (int)(1 + (x * x * x) * (MAX_HISTO_GREEDY - 1));
    int do_greedy;
    if (!HistogramCombineStochastic(image_histo, threshold_size, &do_greedy)) {
      goto Error;
    }
    if (do_greedy && !HistogramCombineGreedy(image_histo)) {
      goto Error;
    }
  }

  // TODO(vrabaud): Optimize HistogramRemap for low-effort compression mode.
  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(orig_histo, image_histo, histogram_symbols);

  ok = 1;

 Error:
  VP8LFreeHistogramSet(orig_histo);
  return ok;
}
