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

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/backward_references_enc.h"
#include "src/enc/histogram_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/utils.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

// Number of partitions for the three dominant (literal, red and blue) symbol
// costs.
#define NUM_PARTITIONS 4
// The size of the bin-hash corresponding to the three dominant costs.
#define BIN_SIZE (NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS)
// Maximum number of histograms allowed in greedy combining algorithm.
#define MAX_HISTO_GREEDY 100

// Enum to meaningfully access the elements of the Histogram arrays.
typedef enum {
  LITERAL = 0,
  RED,
  BLUE,
  ALPHA,
  DISTANCE
} HistogramIndex;

// Return the size of the histogram for a given cache_bits.
static int GetHistogramSize(int cache_bits) {
  const int literal_size = VP8LHistogramNumCodes(cache_bits);
  const size_t total_size = sizeof(VP8LHistogram) + sizeof(int) * literal_size;
  assert(total_size <= (size_t)0x7fffffff);
  return (int)total_size;
}

static void HistogramStatsClear(VP8LHistogram* const h) {
  int i;
  for (i = 0; i < 5; ++i) {
    h->trivial_symbol[i] = VP8L_NON_TRIVIAL_SYM;
    // By default, the histogram is assumed to be used.
    h->is_used[i] = 1;
  }
  h->bit_cost = 0;
  memset(h->costs, 0, sizeof(h->costs));
}

static void HistogramClear(VP8LHistogram* const h) {
  uint32_t* const literal = h->literal;
  const int cache_bits = h->palette_code_bits;
  const int histo_size = GetHistogramSize(cache_bits);
  memset(h, 0, histo_size);
  h->palette_code_bits = cache_bits;
  h->literal = literal;
  HistogramStatsClear(h);
}

// Swap two histogram pointers.
static void HistogramSwap(VP8LHistogram** const h1, VP8LHistogram** const h2) {
  VP8LHistogram* const tmp = *h1;
  *h1 = *h2;
  *h2 = tmp;
}

static void HistogramCopy(const VP8LHistogram* const src,
                          VP8LHistogram* const dst) {
  uint32_t* const dst_literal = dst->literal;
  const int dst_cache_bits = dst->palette_code_bits;
  const int literal_size = VP8LHistogramNumCodes(dst_cache_bits);
  const int histo_size = GetHistogramSize(dst_cache_bits);
  assert(src->palette_code_bits == dst_cache_bits);
  memcpy(dst, src, histo_size);
  dst->literal = dst_literal;
  memcpy(dst->literal, src->literal, literal_size * sizeof(*dst->literal));
}

void VP8LFreeHistogram(VP8LHistogram* const h) { WebPSafeFree(h); }

void VP8LFreeHistogramSet(VP8LHistogramSet* const histograms) {
  WebPSafeFree(histograms);
}

void VP8LHistogramCreate(VP8LHistogram* const h,
                         const VP8LBackwardRefs* const refs,
                         int palette_code_bits) {
  if (palette_code_bits >= 0) {
    h->palette_code_bits = palette_code_bits;
  }
  HistogramClear(h);
  VP8LHistogramStoreRefs(refs, /*distance_modifier=*/NULL,
                         /*distance_modifier_arg0=*/0, h);
}

void VP8LHistogramInit(VP8LHistogram* const h, int palette_code_bits,
                       int init_arrays) {
  h->palette_code_bits = palette_code_bits;
  if (init_arrays) {
    HistogramClear(h);
  } else {
    HistogramStatsClear(h);
  }
}

VP8LHistogram* VP8LAllocateHistogram(int cache_bits) {
  VP8LHistogram* histo = NULL;
  const int total_size = GetHistogramSize(cache_bits);
  uint8_t* const memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;
  histo = (VP8LHistogram*)memory;
  // 'literal' won't necessary be aligned.
  histo->literal = (uint32_t*)(memory + sizeof(VP8LHistogram));
  VP8LHistogramInit(histo, cache_bits, /*init_arrays=*/ 0);
  return histo;
}

// Resets the pointers of the histograms to point to the bit buffer in the set.
static void HistogramSetResetPointers(VP8LHistogramSet* const set,
                                      int cache_bits) {
  int i;
  const int histo_size = GetHistogramSize(cache_bits);
  uint8_t* memory = (uint8_t*) (set->histograms);
  memory += set->max_size * sizeof(*set->histograms);
  for (i = 0; i < set->max_size; ++i) {
    memory = (uint8_t*) WEBP_ALIGN(memory);
    set->histograms[i] = (VP8LHistogram*) memory;
    // 'literal' won't necessary be aligned.
    set->histograms[i]->literal = (uint32_t*)(memory + sizeof(VP8LHistogram));
    memory += histo_size;
  }
}

// Returns the total size of the VP8LHistogramSet.
static size_t HistogramSetTotalSize(int size, int cache_bits) {
  const int histo_size = GetHistogramSize(cache_bits);
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
  const int cache_bits = set->histograms[0]->palette_code_bits;
  const int size = set->max_size;
  const size_t total_size = HistogramSetTotalSize(size, cache_bits);
  uint8_t* memory = (uint8_t*)set;

  memset(memory, 0, total_size);
  memory += sizeof(*set);
  set->histograms = (VP8LHistogram**)memory;
  set->max_size = size;
  set->size = size;
  HistogramSetResetPointers(set, cache_bits);
  for (i = 0; i < size; ++i) {
    set->histograms[i]->palette_code_bits = cache_bits;
  }
}

// Removes the histogram 'i' from 'set'.
static void HistogramSetRemoveHistogram(VP8LHistogramSet* const set, int i) {
  set->histograms[i] = set->histograms[set->size - 1];
  --set->size;
  assert(set->size > 0);
}

// -----------------------------------------------------------------------------

static void HistogramAddSinglePixOrCopy(
    VP8LHistogram* const histo, const PixOrCopy* const v,
    int (*const distance_modifier)(int, int), int distance_modifier_arg0) {
  if (PixOrCopyIsLiteral(v)) {
    ++histo->alpha[PixOrCopyLiteral(v, 3)];
    ++histo->red[PixOrCopyLiteral(v, 2)];
    ++histo->literal[PixOrCopyLiteral(v, 1)];
    ++histo->blue[PixOrCopyLiteral(v, 0)];
  } else if (PixOrCopyIsCacheIdx(v)) {
    const int literal_ix =
        NUM_LITERAL_CODES + NUM_LENGTH_CODES + PixOrCopyCacheIdx(v);
    assert(histo->palette_code_bits != 0);
    ++histo->literal[literal_ix];
  } else {
    int code, extra_bits;
    VP8LPrefixEncodeBits(PixOrCopyLength(v), &code, &extra_bits);
    ++histo->literal[NUM_LITERAL_CODES + code];
    if (distance_modifier == NULL) {
      VP8LPrefixEncodeBits(PixOrCopyDistance(v), &code, &extra_bits);
    } else {
      VP8LPrefixEncodeBits(
          distance_modifier(distance_modifier_arg0, PixOrCopyDistance(v)),
          &code, &extra_bits);
    }
    ++histo->distance[code];
  }
}

void VP8LHistogramStoreRefs(const VP8LBackwardRefs* const refs,
                            int (*const distance_modifier)(int, int),
                            int distance_modifier_arg0,
                            VP8LHistogram* const histo) {
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    HistogramAddSinglePixOrCopy(histo, c.cur_pos, distance_modifier,
                                distance_modifier_arg0);
    VP8LRefsCursorNext(&c);
  }
}

// -----------------------------------------------------------------------------
// Entropy-related functions.

static WEBP_INLINE uint64_t BitsEntropyRefine(const VP8LBitEntropy* entropy) {
  uint64_t mix;
  if (entropy->nonzeros < 5) {
    if (entropy->nonzeros <= 1) {
      return 0;
    }
    // Two symbols, they will be 0 and 1 in a Huffman code.
    // Let's mix in a bit of entropy to favor good clustering when
    // distributions of these are combined.
    if (entropy->nonzeros == 2) {
      return DivRound(99 * ((uint64_t)entropy->sum << LOG_2_PRECISION_BITS) +
                          entropy->entropy,
                      100);
    }
    // No matter what the entropy says, we cannot be better than min_limit
    // with Huffman coding. I am mixing a bit of entropy into the
    // min_limit since it produces much better (~0.5 %) compression results
    // perhaps because of better entropy clustering.
    if (entropy->nonzeros == 3) {
      mix = 950;
    } else {
      mix = 700;  // nonzeros == 4.
    }
  } else {
    mix = 627;
  }

  {
    uint64_t min_limit = (uint64_t)(2 * entropy->sum - entropy->max_val)
                         << LOG_2_PRECISION_BITS;
    min_limit =
        DivRound(mix * min_limit + (1000 - mix) * entropy->entropy, 1000);
    return (entropy->entropy < min_limit) ? min_limit : entropy->entropy;
  }
}

uint64_t VP8LBitsEntropy(const uint32_t* const array, int n) {
  VP8LBitEntropy entropy;
  VP8LBitsEntropyUnrefined(array, n, &entropy);

  return BitsEntropyRefine(&entropy);
}

static uint64_t InitialHuffmanCost(void) {
  // Small bias because Huffman code length is typically not stored in
  // full length.
  static const uint64_t kHuffmanCodeOfHuffmanCodeSize = CODE_LENGTH_CODES * 3;
  // Subtract a bias of 9.1.
  return (kHuffmanCodeOfHuffmanCodeSize << LOG_2_PRECISION_BITS) -
         DivRound(91ll << LOG_2_PRECISION_BITS, 10);
}

// Finalize the Huffman cost based on streak numbers and length type (<3 or >=3)
static uint64_t FinalHuffmanCost(const VP8LStreaks* const stats) {
  // The constants in this function are empirical and got rounded from
  // their original values in 1/8 when switched to 1/1024.
  uint64_t retval = InitialHuffmanCost();
  // Second coefficient: Many zeros in the histogram are covered efficiently
  // by a run-length encode. Originally 2/8.
  uint32_t retval_extra = stats->counts[0] * 1600 + 240 * stats->streaks[0][1];
  // Second coefficient: Constant values are encoded less efficiently, but still
  // RLE'ed. Originally 6/8.
  retval_extra += stats->counts[1] * 2640 + 720 * stats->streaks[1][1];
  // 0s are usually encoded more efficiently than non-0s.
  // Originally 15/8.
  retval_extra += 1840 * stats->streaks[0][0];
  // Originally 26/8.
  retval_extra += 3360 * stats->streaks[1][0];
  return retval + ((uint64_t)retval_extra << (LOG_2_PRECISION_BITS - 10));
}

// Get the symbol entropy for the distribution 'population'.
// Set 'trivial_sym', if there's only one symbol present in the distribution.
static uint64_t PopulationCost(const uint32_t* const population, int length,
                               uint16_t* const trivial_sym,
                               uint8_t* const is_used) {
  VP8LBitEntropy bit_entropy;
  VP8LStreaks stats;
  VP8LGetEntropyUnrefined(population, length, &bit_entropy, &stats);
  if (trivial_sym != NULL) {
    *trivial_sym = (bit_entropy.nonzeros == 1) ? bit_entropy.nonzero_code
                                               : VP8L_NON_TRIVIAL_SYM;
  }
  if (is_used != NULL) {
    // The histogram is used if there is at least one non-zero streak.
    *is_used = (stats.streaks[1][0] != 0 || stats.streaks[1][1] != 0);
  }

  return BitsEntropyRefine(&bit_entropy) + FinalHuffmanCost(&stats);
}

static WEBP_INLINE void GetPopulationInfo(const VP8LHistogram* const histo,
                                          HistogramIndex index,
                                          const uint32_t** population,
                                          int* length) {
  switch (index) {
    case LITERAL:
      *population = histo->literal;
      *length = VP8LHistogramNumCodes(histo->palette_code_bits);
      break;
    case RED:
      *population = histo->red;
      *length = NUM_LITERAL_CODES;
      break;
    case BLUE:
      *population = histo->blue;
      *length = NUM_LITERAL_CODES;
      break;
    case ALPHA:
      *population = histo->alpha;
      *length = NUM_LITERAL_CODES;
      break;
    case DISTANCE:
      *population = histo->distance;
      *length = NUM_DISTANCE_CODES;
      break;
  }
}

// trivial_at_end is 1 if the two histograms only have one element that is
// non-zero: both the zero-th one, or both the last one.
// 'index' is the index of the symbol in the histogram (literal, red, blue,
// alpha, distance).
static WEBP_INLINE uint64_t GetCombinedEntropy(const VP8LHistogram* const h1,
                                               const VP8LHistogram* const h2,
                                               HistogramIndex index) {
  const uint32_t* X;
  const uint32_t* Y;
  int length;
  VP8LStreaks stats;
  VP8LBitEntropy bit_entropy;
  const int is_h1_used = h1->is_used[index];
  const int is_h2_used = h2->is_used[index];
  const int is_trivial = h1->trivial_symbol[index] != VP8L_NON_TRIVIAL_SYM &&
                         h1->trivial_symbol[index] == h2->trivial_symbol[index];

  if (is_trivial || !is_h1_used || !is_h2_used) {
    if (is_h1_used) return h1->costs[index];
    return h2->costs[index];
  }
  assert(is_h1_used && is_h2_used);

  GetPopulationInfo(h1, index, &X, &length);
  GetPopulationInfo(h2, index, &Y, &length);
  VP8LGetCombinedEntropyUnrefined(X, Y, length, &bit_entropy, &stats);
  return BitsEntropyRefine(&bit_entropy) + FinalHuffmanCost(&stats);
}

// Estimates the Entropy + Huffman + other block overhead size cost.
uint64_t VP8LHistogramEstimateBits(const VP8LHistogram* const h) {
  int i;
  uint64_t cost = 0;
  for (i = 0; i < 5; ++i) {
    int length;
    const uint32_t* population;
    GetPopulationInfo(h, (HistogramIndex)i, &population, &length);
    cost += PopulationCost(population, length, /*trivial_sym=*/NULL,
                           /*is_used=*/NULL);
  }
  cost += ((uint64_t)(VP8LExtraCost(h->literal + NUM_LITERAL_CODES,
                                    NUM_LENGTH_CODES) +
                      VP8LExtraCost(h->distance, NUM_DISTANCE_CODES))
           << LOG_2_PRECISION_BITS);
  return cost;
}

// -----------------------------------------------------------------------------
// Various histogram combine/cost-eval functions

// Set a + b in b, saturating at WEBP_INT64_MAX.
static WEBP_INLINE void SaturateAdd(uint64_t a, int64_t* b) {
  if (*b < 0 || (int64_t)a <= WEBP_INT64_MAX - *b) {
    *b += (int64_t)a;
  } else {
    *b = WEBP_INT64_MAX;
  }
}

// Returns 1 if the cost of the combined histogram is less than the threshold.
// Otherwise returns 0 and the cost is invalid due to early bail-out.
WEBP_NODISCARD static int GetCombinedHistogramEntropy(
    const VP8LHistogram* const a, const VP8LHistogram* const b,
    int64_t cost_threshold_in, uint64_t* cost, uint64_t costs[5]) {
  int i;
  const uint64_t cost_threshold = (uint64_t)cost_threshold_in;
  assert(a->palette_code_bits == b->palette_code_bits);
  if (cost_threshold_in <= 0) return 0;
  *cost = 0;

  // No need to add the extra cost for length and distance as it is a constant
  // that does not influence the histograms.
  for (i = 0; i < 5; ++i) {
    costs[i] = GetCombinedEntropy(a, b, (HistogramIndex)i);
    *cost += costs[i];
    if (*cost >= cost_threshold) return 0;
  }

  return 1;
}

static WEBP_INLINE void HistogramAdd(const VP8LHistogram* const h1,
                                     const VP8LHistogram* const h2,
                                     VP8LHistogram* const hout) {
  int i;
  assert(h1->palette_code_bits == h2->palette_code_bits);

  for (i = 0; i < 5; ++i) {
    int length;
    const uint32_t *p1, *p2, *pout_const;
    uint32_t* pout;
    GetPopulationInfo(h1, (HistogramIndex)i, &p1, &length);
    GetPopulationInfo(h2, (HistogramIndex)i, &p2, &length);
    GetPopulationInfo(hout, (HistogramIndex)i, &pout_const, &length);
    pout = (uint32_t*)pout_const;
    if (h2 == hout) {
      if (h1->is_used[i]) {
        if (hout->is_used[i]) {
          VP8LAddVectorEq(p1, pout, length);
        } else {
          memcpy(pout, p1, length * sizeof(pout[0]));
        }
      }
    } else {
      if (h1->is_used[i]) {
        if (h2->is_used[i]) {
          VP8LAddVector(p1, p2, pout, length);
        } else {
          memcpy(pout, p1, length * sizeof(pout[0]));
        }
      } else if (h2->is_used[i]) {
        memcpy(pout, p2, length * sizeof(pout[0]));
      } else {
        memset(pout, 0, length * sizeof(pout[0]));
      }
    }
  }

  for (i = 0; i < 5; ++i) {
    hout->trivial_symbol[i] = h1->trivial_symbol[i] == h2->trivial_symbol[i]
                                  ? h1->trivial_symbol[i]
                                  : VP8L_NON_TRIVIAL_SYM;
    hout->is_used[i] = h1->is_used[i] || h2->is_used[i];
  }
}

static void UpdateHistogramCost(uint64_t bit_cost, uint64_t costs[5],
                                VP8LHistogram* const h) {
  int i;
  h->bit_cost = bit_cost;
  for (i = 0; i < 5; ++i) {
    h->costs[i] = costs[i];
  }
}

// Performs out = a + b, computing the cost C(a+b) - C(a) - C(b) while comparing
// to the threshold value 'cost_threshold'. The score returned is
//  Score = C(a+b) - C(a) - C(b), where C(a) + C(b) is known and fixed.
// Since the previous score passed is 'cost_threshold', we only need to compare
// the partial cost against 'cost_threshold + C(a) + C(b)' to possibly bail-out
// early.
// Returns 1 if the cost is less than the threshold.
// Otherwise returns 0 and the cost is invalid due to early bail-out.
WEBP_NODISCARD static int HistogramAddEval(const VP8LHistogram* const a,
                                           const VP8LHistogram* const b,
                                           VP8LHistogram* const out,
                                           int64_t cost_threshold) {
  const uint64_t sum_cost = a->bit_cost + b->bit_cost;
  uint64_t bit_cost, costs[5];
  SaturateAdd(sum_cost, &cost_threshold);
  if (!GetCombinedHistogramEntropy(a, b, cost_threshold, &bit_cost, costs)) {
    return 0;
  }

  HistogramAdd(a, b, out);
  UpdateHistogramCost(bit_cost, costs, out);
  return 1;
}

// Same as HistogramAddEval(), except that the resulting histogram
// is not stored. Only the cost C(a+b) - C(a) is evaluated. We omit
// the term C(b) which is constant over all the evaluations.
// Returns 1 if the cost is less than the threshold.
// Otherwise returns 0 and the cost is invalid due to early bail-out.
WEBP_NODISCARD static int HistogramAddThresh(const VP8LHistogram* const a,
                                             const VP8LHistogram* const b,
                                             int64_t cost_threshold,
                                             int64_t* cost_out) {
  uint64_t cost, costs[5];
  assert(a != NULL && b != NULL);
  SaturateAdd(a->bit_cost, &cost_threshold);
  if (!GetCombinedHistogramEntropy(a, b, cost_threshold, &cost, costs)) {
    return 0;
  }

  *cost_out = (int64_t)cost - (int64_t)a->bit_cost;
  return 1;
}

// -----------------------------------------------------------------------------

// The structure to keep track of cost range for the three dominant entropy
// symbols.
typedef struct {
  uint64_t literal_max;
  uint64_t literal_min;
  uint64_t red_max;
  uint64_t red_min;
  uint64_t blue_max;
  uint64_t blue_min;
} DominantCostRange;

static void DominantCostRangeInit(DominantCostRange* const c) {
  c->literal_max = 0;
  c->literal_min = WEBP_UINT64_MAX;
  c->red_max = 0;
  c->red_min = WEBP_UINT64_MAX;
  c->blue_max = 0;
  c->blue_min = WEBP_UINT64_MAX;
}

static void UpdateDominantCostRange(
    const VP8LHistogram* const h, DominantCostRange* const c) {
  if (c->literal_max < h->costs[LITERAL]) c->literal_max = h->costs[LITERAL];
  if (c->literal_min > h->costs[LITERAL]) c->literal_min = h->costs[LITERAL];
  if (c->red_max < h->costs[RED]) c->red_max = h->costs[RED];
  if (c->red_min > h->costs[RED]) c->red_min = h->costs[RED];
  if (c->blue_max < h->costs[BLUE]) c->blue_max = h->costs[BLUE];
  if (c->blue_min > h->costs[BLUE]) c->blue_min = h->costs[BLUE];
}

static void ComputeHistogramCost(VP8LHistogram* const h) {
  int i;
  // No need to add the extra cost for length and distance as it is a constant
  // that does not influence the histograms.
  for (i = 0; i < 5; ++i) {
    const uint32_t* population;
    int length;
    GetPopulationInfo(h, i, &population, &length);
    h->costs[i] = PopulationCost(population, length, &h->trivial_symbol[i],
                                 &h->is_used[i]);
  }
  h->bit_cost = h->costs[LITERAL] + h->costs[RED] + h->costs[BLUE] +
                h->costs[ALPHA] + h->costs[DISTANCE];
}

static int GetBinIdForEntropy(uint64_t min, uint64_t max, uint64_t val) {
  const uint64_t range = max - min;
  if (range > 0) {
    const uint64_t delta = val - min;
    return (int)((NUM_PARTITIONS - 1e-6) * delta / range);
  } else {
    return 0;
  }
}

static int GetHistoBinIndex(const VP8LHistogram* const h,
                            const DominantCostRange* const c, int low_effort) {
  int bin_id =
      GetBinIdForEntropy(c->literal_min, c->literal_max, h->costs[LITERAL]);
  assert(bin_id < NUM_PARTITIONS);
  if (!low_effort) {
    bin_id = bin_id * NUM_PARTITIONS +
             GetBinIdForEntropy(c->red_min, c->red_max, h->costs[RED]);
    bin_id = bin_id * NUM_PARTITIONS +
             GetBinIdForEntropy(c->blue_min, c->blue_max, h->costs[BLUE]);
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
    HistogramAddSinglePixOrCopy(histograms[ix], v, NULL, 0);
    x += PixOrCopyLength(v);
    while (x >= xsize) {
      x -= xsize;
      ++y;
    }
    VP8LRefsCursorNext(&c);
  }
}

// Copies the histograms and computes its bit_cost.
static void HistogramCopyAndAnalyze(VP8LHistogramSet* const orig_histo,
                                    VP8LHistogramSet* const image_histo) {
  int i;
  VP8LHistogram** const orig_histograms = orig_histo->histograms;
  VP8LHistogram** const histograms = image_histo->histograms;
  assert(image_histo->max_size == orig_histo->max_size);
  image_histo->size = 0;
  for (i = 0; i < orig_histo->max_size; ++i) {
    VP8LHistogram* const histo = orig_histograms[i];
    ComputeHistogramCost(histo);

    // Skip the histogram if it is completely empty, which can happen for tiles
    // with no information (when they are skipped because of LZ77).
    if (!histo->is_used[LITERAL] && !histo->is_used[RED] &&
        !histo->is_used[BLUE] && !histo->is_used[ALPHA] &&
        !histo->is_used[DISTANCE]) {
      // The first histogram is always used.
      assert(i > 0);
      orig_histograms[i] = NULL;
    } else {
      // Copy histograms from orig_histo[] to image_histo[].
      HistogramCopy(histo, histograms[image_histo->size]);
      ++image_histo->size;
    }
  }
}

// Partition histograms to different entropy bins for three dominant (literal,
// red and blue) symbol costs and compute the histogram aggregate bit_cost.
static void HistogramAnalyzeEntropyBin(VP8LHistogramSet* const image_histo,
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
    histograms[i]->bin_id =
        GetHistoBinIndex(histograms[i], &cost_range, low_effort);
  }
}

// Merges some histograms with same bin_id together if it's advantageous.
// Sets the remaining histograms to NULL.
// 'combine_cost_factor' has to be divided by 100.
static void HistogramCombineEntropyBin(VP8LHistogramSet* const image_histo,
                                       VP8LHistogram* cur_combo, int num_bins,
                                       int32_t combine_cost_factor,
                                       int low_effort) {
  VP8LHistogram** const histograms = image_histo->histograms;
  int idx;
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

  for (idx = 0; idx < image_histo->size;) {
    const int bin_id = histograms[idx]->bin_id;
    const int first = bin_info[bin_id].first;
    if (first == -1) {
      bin_info[bin_id].first = idx;
      ++idx;
    } else if (low_effort) {
      HistogramAdd(histograms[idx], histograms[first], histograms[first]);
      HistogramSetRemoveHistogram(image_histo, idx);
    } else {
      // try to merge #idx into #first (both share the same bin_id)
      const uint64_t bit_cost = histograms[idx]->bit_cost;
      const int64_t bit_cost_thresh =
          -DivRound((int64_t)bit_cost * combine_cost_factor, 100);
      if (HistogramAddEval(histograms[first], histograms[idx], cur_combo,
                           bit_cost_thresh)) {
        const int max_combine_failures = 32;
        // Try to merge two histograms only if the combo is a trivial one or
        // the two candidate histograms are already non-trivial.
        // For some images, 'try_combine' turns out to be false for a lot of
        // histogram pairs. In that case, we fallback to combining
        // histograms as usual to avoid increasing the header size.
        int try_combine =
            cur_combo->trivial_symbol[RED] != VP8L_NON_TRIVIAL_SYM &&
            cur_combo->trivial_symbol[BLUE] != VP8L_NON_TRIVIAL_SYM &&
            cur_combo->trivial_symbol[ALPHA] != VP8L_NON_TRIVIAL_SYM;
        if (!try_combine) {
          try_combine =
              histograms[idx]->trivial_symbol[RED] == VP8L_NON_TRIVIAL_SYM ||
              histograms[idx]->trivial_symbol[BLUE] == VP8L_NON_TRIVIAL_SYM ||
              histograms[idx]->trivial_symbol[ALPHA] == VP8L_NON_TRIVIAL_SYM;
          try_combine &=
              histograms[first]->trivial_symbol[RED] == VP8L_NON_TRIVIAL_SYM ||
              histograms[first]->trivial_symbol[BLUE] == VP8L_NON_TRIVIAL_SYM ||
              histograms[first]->trivial_symbol[ALPHA] == VP8L_NON_TRIVIAL_SYM;
        }
        if (try_combine ||
            bin_info[bin_id].num_combine_failures >= max_combine_failures) {
          // move the (better) merged histogram to its final slot
          HistogramSwap(&cur_combo, &histograms[first]);
          HistogramSetRemoveHistogram(image_histo, idx);
        } else {
          ++bin_info[bin_id].num_combine_failures;
          ++idx;
        }
      } else {
        ++idx;
      }
    }
  }
  if (low_effort) {
    // for low_effort case, update the final cost when everything is merged
    for (idx = 0; idx < image_histo->size; ++idx) {
      ComputeHistogramCost(histograms[idx]);
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
  int64_t cost_diff;
  uint64_t cost_combo;
  uint64_t costs[5];
} HistogramPair;

typedef struct {
  HistogramPair* queue;
  int size;
  int max_size;
} HistoQueue;

static int HistoQueueInit(HistoQueue* const histo_queue, const int max_size) {
  histo_queue->size = 0;
  histo_queue->max_size = max_size;
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
  assert(pair->cost_diff < 0);
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

// Replaces the bad_id with good_id in the pair.
static void HistoQueueFixPair(int bad_id, int good_id,
                              HistogramPair* const pair) {
  if (pair->idx1 == bad_id) pair->idx1 = good_id;
  if (pair->idx2 == bad_id) pair->idx2 = good_id;
  if (pair->idx1 > pair->idx2) {
    const int tmp = pair->idx1;
    pair->idx1 = pair->idx2;
    pair->idx2 = tmp;
  }
}

// Update the cost diff and combo of a pair of histograms. This needs to be
// called when the histograms have been merged with a third one.
// Returns 1 if the cost diff is less than the threshold.
// Otherwise returns 0 and the cost is invalid due to early bail-out.
WEBP_NODISCARD static int HistoQueueUpdatePair(const VP8LHistogram* const h1,
                                               const VP8LHistogram* const h2,
                                               int64_t cost_threshold,
                                               HistogramPair* const pair) {
  const int64_t sum_cost = h1->bit_cost + h2->bit_cost;
  SaturateAdd(sum_cost, &cost_threshold);
  if (!GetCombinedHistogramEntropy(h1, h2, cost_threshold, &pair->cost_combo,
                                   pair->costs)) {
    return 0;
  }
  pair->cost_diff = (int64_t)pair->cost_combo - sum_cost;
  return 1;
}

// Create a pair from indices "idx1" and "idx2" provided its cost
// is inferior to "threshold", a negative entropy.
// It returns the cost of the pair, or 0 if it superior to threshold.
static int64_t HistoQueuePush(HistoQueue* const histo_queue,
                              VP8LHistogram** const histograms, int idx1,
                              int idx2, int64_t threshold) {
  const VP8LHistogram* h1;
  const VP8LHistogram* h2;
  HistogramPair pair;

  // Stop here if the queue is full.
  if (histo_queue->size == histo_queue->max_size) return 0;
  assert(threshold <= 0);
  if (idx1 > idx2) {
    const int tmp = idx2;
    idx2 = idx1;
    idx1 = tmp;
  }
  pair.idx1 = idx1;
  pair.idx2 = idx2;
  h1 = histograms[idx1];
  h2 = histograms[idx2];

  // Do not even consider the pair if it does not improve the entropy.
  if (!HistoQueueUpdatePair(h1, h2, threshold, &pair)) return 0;

  histo_queue->queue[histo_queue->size++] = pair;
  HistoQueueUpdateHead(histo_queue, &histo_queue->queue[histo_queue->size - 1]);

  return pair.cost_diff;
}

// -----------------------------------------------------------------------------

// Combines histograms by continuously choosing the one with the highest cost
// reduction.
static int HistogramCombineGreedy(VP8LHistogramSet* const image_histo) {
  int ok = 0;
  const int image_histo_size = image_histo->size;
  int i, j;
  VP8LHistogram** const histograms = image_histo->histograms;
  // Priority queue of histogram pairs.
  HistoQueue histo_queue;

  // image_histo_size^2 for the queue size is safe. If you look at
  // HistogramCombineGreedy, and imagine that UpdateQueueFront always pushes
  // data to the queue, you insert at most:
  // - image_histo_size*(image_histo_size-1)/2 (the first two for loops)
  // - image_histo_size - 1 in the last for loop at the first iteration of
  //   the while loop, image_histo_size - 2 at the second iteration ...
  //   therefore image_histo_size*(image_histo_size-1)/2 overall too
  if (!HistoQueueInit(&histo_queue, image_histo_size * image_histo_size)) {
    goto End;
  }

  // Initialize the queue.
  for (i = 0; i < image_histo_size; ++i) {
    for (j = i + 1; j < image_histo_size; ++j) {
      HistoQueuePush(&histo_queue, histograms, i, j, 0);
    }
  }

  while (histo_queue.size > 0) {
    const int idx1 = histo_queue.queue[0].idx1;
    const int idx2 = histo_queue.queue[0].idx2;
    HistogramAdd(histograms[idx2], histograms[idx1], histograms[idx1]);
    UpdateHistogramCost(histo_queue.queue[0].cost_combo,
                        histo_queue.queue[0].costs, histograms[idx1]);

    // Remove merged histogram.
    HistogramSetRemoveHistogram(image_histo, idx2);

    // Remove pairs intersecting the just combined best pair.
    for (i = 0; i < histo_queue.size;) {
      HistogramPair* const p = histo_queue.queue + i;
      if (p->idx1 == idx1 || p->idx2 == idx1 ||
          p->idx1 == idx2 || p->idx2 == idx2) {
        HistoQueuePopPair(&histo_queue, p);
      } else {
        HistoQueueFixPair(image_histo->size, idx2, p);
        HistoQueueUpdateHead(&histo_queue, p);
        ++i;
      }
    }

    // Push new pairs formed with combined histogram to the queue.
    for (i = 0; i < image_histo->size; ++i) {
      if (i == idx1) continue;
      HistoQueuePush(&histo_queue, image_histo->histograms, idx1, i, 0);
    }
  }

  ok = 1;

 End:
  HistoQueueClear(&histo_queue);
  return ok;
}

// Perform histogram aggregation using a stochastic approach.
// 'do_greedy' is set to 1 if a greedy approach needs to be performed
// afterwards, 0 otherwise.
static int HistogramCombineStochastic(VP8LHistogramSet* const image_histo,
                                      int min_cluster_size,
                                      int* const do_greedy) {
  int j, iter;
  uint32_t seed = 1;
  int tries_with_no_success = 0;
  const int outer_iters = image_histo->size;
  const int num_tries_no_success = outer_iters / 2;
  VP8LHistogram** const histograms = image_histo->histograms;
  // Priority queue of histogram pairs. Its size of 'kHistoQueueSize'
  // impacts the quality of the compression and the speed: the smaller the
  // faster but the worse for the compression.
  HistoQueue histo_queue;
  const int kHistoQueueSize = 9;
  int ok = 0;

  if (image_histo->size < min_cluster_size) {
    *do_greedy = 1;
    return 1;
  }

  if (!HistoQueueInit(&histo_queue, kHistoQueueSize)) goto End;

  // Collapse similar histograms in 'image_histo'.
  for (iter = 0; iter < outer_iters && image_histo->size >= min_cluster_size &&
                ++tries_with_no_success < num_tries_no_success;
      ++iter) {
    int64_t best_cost =
        (histo_queue.size == 0) ? 0 : histo_queue.queue[0].cost_diff;
    int best_idx1 = -1, best_idx2 = 1;
    const uint32_t rand_range = (image_histo->size - 1) * (image_histo->size);
    // (image_histo->size) / 2 was chosen empirically. Less means faster but
    // worse compression.
    const int num_tries = (image_histo->size) / 2;

    // Pick random samples.
    for (j = 0; image_histo->size >= 2 && j < num_tries; ++j) {
      int64_t curr_cost;
      // Choose two different histograms at random and try to combine them.
      const uint32_t tmp = MyRand(&seed) % rand_range;
      uint32_t idx1 = tmp / (image_histo->size - 1);
      uint32_t idx2 = tmp % (image_histo->size - 1);
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

    // Get the best histograms.
    best_idx1 = histo_queue.queue[0].idx1;
    best_idx2 = histo_queue.queue[0].idx2;
    assert(best_idx1 < best_idx2);
    // Merge the histograms and remove best_idx2 from the queue.
    HistogramAdd(histograms[best_idx2], histograms[best_idx1],
                 histograms[best_idx1]);
    UpdateHistogramCost(histo_queue.queue[0].cost_combo,
                        histo_queue.queue[0].costs, histograms[best_idx1]);
    HistogramSetRemoveHistogram(image_histo, best_idx2);
    // Parse the queue and update each pair that deals with best_idx1,
    // best_idx2 or image_histo_size.
    for (j = 0; j < histo_queue.size;) {
      HistogramPair* const p = histo_queue.queue + j;
      const int is_idx1_best = p->idx1 == best_idx1 || p->idx1 == best_idx2;
      const int is_idx2_best = p->idx2 == best_idx1 || p->idx2 == best_idx2;
      // The front pair could have been duplicated by a random pick so
      // check for it all the time nevertheless.
      if (is_idx1_best && is_idx2_best) {
        HistoQueuePopPair(&histo_queue, p);
        continue;
      }
      // Any pair containing one of the two best indices should only refer to
      // best_idx1. Its cost should also be updated.
      if (is_idx1_best || is_idx2_best) {
        HistoQueueFixPair(best_idx2, best_idx1, p);
        // Re-evaluate the cost of an updated pair.
        if (!HistoQueueUpdatePair(histograms[p->idx1], histograms[p->idx2], 0,
                                  p)) {
          HistoQueuePopPair(&histo_queue, p);
          continue;
        }
      }
      HistoQueueFixPair(image_histo->size, best_idx2, p);
      HistoQueueUpdateHead(&histo_queue, p);
      ++j;
    }
    tries_with_no_success = 0;
  }
  *do_greedy = (image_histo->size <= min_cluster_size);
  ok = 1;

 End:
  HistoQueueClear(&histo_queue);
  return ok;
}

// -----------------------------------------------------------------------------
// Histogram refinement

// Find the best 'out' histogram for each of the 'in' histograms.
// At call-time, 'out' contains the histograms of the clusters.
// Note: we assume that out[]->bit_cost is already up-to-date.
static void HistogramRemap(const VP8LHistogramSet* const in,
                           VP8LHistogramSet* const out,
                           uint32_t* const symbols) {
  int i;
  VP8LHistogram** const in_histo = in->histograms;
  VP8LHistogram** const out_histo = out->histograms;
  const int in_size = out->max_size;
  const int out_size = out->size;
  if (out_size > 1) {
    for (i = 0; i < in_size; ++i) {
      int best_out = 0;
      int64_t best_bits = WEBP_INT64_MAX;
      int k;
      if (in_histo[i] == NULL) {
        // Arbitrarily set to the previous value if unused to help future LZ77.
        symbols[i] = symbols[i - 1];
        continue;
      }
      for (k = 0; k < out_size; ++k) {
        int64_t cur_bits;
        if (HistogramAddThresh(out_histo[k], in_histo[i], best_bits,
                               &cur_bits)) {
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
  VP8LHistogramSetClear(out);
  out->size = out_size;

  for (i = 0; i < in_size; ++i) {
    int idx;
    if (in_histo[i] == NULL) continue;
    idx = symbols[i];
    HistogramAdd(in_histo[i], out_histo[idx], out_histo[idx]);
  }
}

static int32_t GetCombineCostFactor(int histo_size, int quality) {
  int32_t combine_cost_factor = 16;
  if (quality < 90) {
    if (histo_size > 256) combine_cost_factor /= 2;
    if (histo_size > 512) combine_cost_factor /= 2;
    if (histo_size > 1024) combine_cost_factor /= 2;
    if (quality <= 50) combine_cost_factor /= 2;
  }
  return combine_cost_factor;
}

int VP8LGetHistoImageSymbols(int xsize, int ysize,
                             const VP8LBackwardRefs* const refs, int quality,
                             int low_effort, int histogram_bits, int cache_bits,
                             VP8LHistogramSet* const image_histo,
                             VP8LHistogram* const tmp_histo,
                             uint32_t* const histogram_symbols,
                             const WebPPicture* const pic, int percent_range,
                             int* const percent) {
  const int histo_xsize =
      histogram_bits ? VP8LSubSampleSize(xsize, histogram_bits) : 1;
  const int histo_ysize =
      histogram_bits ? VP8LSubSampleSize(ysize, histogram_bits) : 1;
  const int image_histo_raw_size = histo_xsize * histo_ysize;
  VP8LHistogramSet* const orig_histo =
      VP8LAllocateHistogramSet(image_histo_raw_size, cache_bits);
  // Don't attempt linear bin-partition heuristic for
  // histograms of small sizes (as bin_map will be very sparse) and
  // maximum quality q==100 (to preserve the compression gains at that level).
  const int entropy_combine_num_bins = low_effort ? NUM_PARTITIONS : BIN_SIZE;
  int entropy_combine;
  if (orig_histo == NULL) {
    WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
    goto Error;
  }

  // Construct the histograms from backward references.
  HistogramBuild(xsize, histogram_bits, refs, orig_histo);
  HistogramCopyAndAnalyze(orig_histo, image_histo);
  entropy_combine =
      (image_histo->size > entropy_combine_num_bins * 2) && (quality < 100);

  if (entropy_combine) {
    const int32_t combine_cost_factor =
        GetCombineCostFactor(image_histo_raw_size, quality);

    HistogramAnalyzeEntropyBin(image_histo, low_effort);
    // Collapse histograms with similar entropy.
    HistogramCombineEntropyBin(image_histo, tmp_histo, entropy_combine_num_bins,
                               combine_cost_factor, low_effort);
  }

  // Don't combine the histograms using stochastic and greedy heuristics for
  // low-effort compression mode.
  if (!low_effort || !entropy_combine) {
    // cubic ramp between 1 and MAX_HISTO_GREEDY:
    const int threshold_size =
        (int)(1 + DivRound(quality * quality * quality * (MAX_HISTO_GREEDY - 1),
                           100 * 100 * 100));
    int do_greedy;
    if (!HistogramCombineStochastic(image_histo, threshold_size, &do_greedy)) {
      WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
      goto Error;
    }
    if (do_greedy) {
      if (!HistogramCombineGreedy(image_histo)) {
        WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
        goto Error;
      }
    }
  }

  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(orig_histo, image_histo, histogram_symbols);

  if (!WebPReportProgress(pic, *percent + percent_range, percent)) {
    goto Error;
  }

 Error:
  VP8LFreeHistogramSet(orig_histo);
  return (pic->error_code == VP8_ENC_OK);
}
