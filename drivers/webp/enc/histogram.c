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
#include "webp/config.h"
#endif

#include <math.h>

#include "./backward_references.h"
#include "./histogram.h"
#include "../dsp/lossless.h"
#include "../utils/utils.h"

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
  const int histo_size = VP8LGetHistogramSize(dst_cache_bits);
  assert(src->palette_code_bits_ == dst_cache_bits);
  memcpy(dst, src, histo_size);
  dst->literal_ = dst_literal;
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
    VP8LHistogramAddSinglePixOrCopy(histo, c.cur_pos);
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

void VP8LHistogramInit(VP8LHistogram* const p, int palette_code_bits) {
  p->palette_code_bits_ = palette_code_bits;
  HistogramClear(p);
}

VP8LHistogram* VP8LAllocateHistogram(int cache_bits) {
  VP8LHistogram* histo = NULL;
  const int total_size = VP8LGetHistogramSize(cache_bits);
  uint8_t* const memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;
  histo = (VP8LHistogram*)memory;
  // literal_ won't necessary be aligned.
  histo->literal_ = (uint32_t*)(memory + sizeof(VP8LHistogram));
  VP8LHistogramInit(histo, cache_bits);
  return histo;
}

VP8LHistogramSet* VP8LAllocateHistogramSet(int size, int cache_bits) {
  int i;
  VP8LHistogramSet* set;
  const int histo_size = VP8LGetHistogramSize(cache_bits);
  const size_t total_size =
      sizeof(*set) + size * (sizeof(*set->histograms) +
      histo_size + WEBP_ALIGN_CST);
  uint8_t* memory = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*memory));
  if (memory == NULL) return NULL;

  set = (VP8LHistogramSet*)memory;
  memory += sizeof(*set);
  set->histograms = (VP8LHistogram**)memory;
  memory += size * sizeof(*set->histograms);
  set->max_size = size;
  set->size = size;
  for (i = 0; i < size; ++i) {
    memory = (uint8_t*)WEBP_ALIGN(memory);
    set->histograms[i] = (VP8LHistogram*)memory;
    // literal_ won't necessary be aligned.
    set->histograms[i]->literal_ = (uint32_t*)(memory + sizeof(VP8LHistogram));
    VP8LHistogramInit(set->histograms[i], cache_bits);
    memory += histo_size;
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
    const int literal_ix =
        NUM_LITERAL_CODES + NUM_LENGTH_CODES + PixOrCopyCacheIdx(v);
    ++histo->literal_[literal_ix];
  } else {
    int code, extra_bits;
    VP8LPrefixEncodeBits(PixOrCopyLength(v), &code, &extra_bits);
    ++histo->literal_[NUM_LITERAL_CODES + code];
    VP8LPrefixEncodeBits(PixOrCopyDistance(v), &code, &extra_bits);
    ++histo->distance_[code];
  }
}

// -----------------------------------------------------------------------------
// Various histogram combine/cost-eval functions

static int GetCombinedHistogramEntropy(const VP8LHistogram* const a,
                                       const VP8LHistogram* const b,
                                       double cost_threshold,
                                       double* cost) {
  const int palette_code_bits = a->palette_code_bits_;
  assert(a->palette_code_bits_ == b->palette_code_bits_);
  *cost += VP8LGetCombinedEntropy(a->literal_, b->literal_,
                                  VP8LHistogramNumCodes(palette_code_bits));
  *cost += VP8LExtraCostCombined(a->literal_ + NUM_LITERAL_CODES,
                                 b->literal_ + NUM_LITERAL_CODES,
                                 NUM_LENGTH_CODES);
  if (*cost > cost_threshold) return 0;

  *cost += VP8LGetCombinedEntropy(a->red_, b->red_, NUM_LITERAL_CODES);
  if (*cost > cost_threshold) return 0;

  *cost += VP8LGetCombinedEntropy(a->blue_, b->blue_, NUM_LITERAL_CODES);
  if (*cost > cost_threshold) return 0;

  *cost += VP8LGetCombinedEntropy(a->alpha_, b->alpha_, NUM_LITERAL_CODES);
  if (*cost > cost_threshold) return 0;

  *cost += VP8LGetCombinedEntropy(a->distance_, b->distance_,
                                  NUM_DISTANCE_CODES);
  *cost += VP8LExtraCostCombined(a->distance_, b->distance_,
                                 NUM_DISTANCE_CODES);
  if (*cost > cost_threshold) return 0;

  return 1;
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
    VP8LHistogramAdd(a, b, out);
    out->bit_cost_ = cost;
    out->palette_code_bits_ = a->palette_code_bits_;
    out->trivial_symbol_ = (a->trivial_symbol_ == b->trivial_symbol_) ?
        a->trivial_symbol_ : VP8L_NON_TRIVIAL_SYM;
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
  const double alpha_cost = VP8LPopulationCost(h->alpha_, NUM_LITERAL_CODES,
                                               &alpha_sym);
  const double distance_cost =
      VP8LPopulationCost(h->distance_, NUM_DISTANCE_CODES, NULL) +
      VP8LExtraCost(h->distance_, NUM_DISTANCE_CODES);
  const int num_codes = VP8LHistogramNumCodes(h->palette_code_bits_);
  h->literal_cost_ = VP8LPopulationCost(h->literal_, num_codes, NULL) +
                     VP8LExtraCost(h->literal_ + NUM_LITERAL_CODES,
                                   NUM_LENGTH_CODES);
  h->red_cost_ = VP8LPopulationCost(h->red_, NUM_LITERAL_CODES, &red_sym);
  h->blue_cost_ = VP8LPopulationCost(h->blue_, NUM_LITERAL_CODES, &blue_sym);
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
  const double range = max - min + 1e-6;
  const double delta = val - min;
  return (int)(NUM_PARTITIONS * delta / range);
}

static int GetHistoBinIndexLowEffort(
    const VP8LHistogram* const h, const DominantCostRange* const c) {
  const int bin_id = GetBinIdForEntropy(c->literal_min_, c->literal_max_,
                                        h->literal_cost_);
  assert(bin_id < NUM_PARTITIONS);
  return bin_id;
}

static int GetHistoBinIndex(
    const VP8LHistogram* const h, const DominantCostRange* const c) {
  const int bin_id =
      GetBinIdForEntropy(c->blue_min_, c->blue_max_, h->blue_cost_) +
      NUM_PARTITIONS * GetBinIdForEntropy(c->red_min_, c->red_max_,
                                          h->red_cost_) +
      NUM_PARTITIONS * NUM_PARTITIONS * GetBinIdForEntropy(c->literal_min_,
                                                           c->literal_max_,
                                                           h->literal_cost_);
  assert(bin_id < BIN_SIZE);
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
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    const int ix = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
    VP8LHistogramAddSinglePixOrCopy(histograms[ix], v);
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
  for (i = 0; i < histo_size; ++i) {
    VP8LHistogram* const histo = orig_histograms[i];
    UpdateHistogramCost(histo);
    // Copy histograms from orig_histo[] to image_histo[].
    HistogramCopy(histo, histograms[i]);
  }
}

// Partition histograms to different entropy bins for three dominant (literal,
// red and blue) symbol costs and compute the histogram aggregate bit_cost.
static void HistogramAnalyzeEntropyBin(VP8LHistogramSet* const image_histo,
                                       int16_t* const bin_map, int low_effort) {
  int i;
  VP8LHistogram** const histograms = image_histo->histograms;
  const int histo_size = image_histo->size;
  const int bin_depth = histo_size + 1;
  DominantCostRange cost_range;
  DominantCostRangeInit(&cost_range);

  // Analyze the dominant (literal, red and blue) entropy costs.
  for (i = 0; i < histo_size; ++i) {
    VP8LHistogram* const histo = histograms[i];
    UpdateDominantCostRange(histo, &cost_range);
  }

  // bin-hash histograms on three of the dominant (literal, red and blue)
  // symbol costs.
  for (i = 0; i < histo_size; ++i) {
    int num_histos;
    VP8LHistogram* const histo = histograms[i];
    const int16_t bin_id = low_effort ?
        (int16_t)GetHistoBinIndexLowEffort(histo, &cost_range) :
        (int16_t)GetHistoBinIndex(histo, &cost_range);
    const int bin_offset = bin_id * bin_depth;
    // bin_map[n][0] for every bin 'n' maintains the counter for the number of
    // histograms in that bin.
    // Get and increment the num_histos in that bin.
    num_histos = ++bin_map[bin_offset];
    assert(bin_offset + num_histos < bin_depth * BIN_SIZE);
    // Add histogram i'th index at num_histos (last) position in the bin_map.
    bin_map[bin_offset + num_histos] = i;
  }
}

// Compact the histogram set by removing unused entries.
static void HistogramCompactBins(VP8LHistogramSet* const image_histo) {
  VP8LHistogram** const histograms = image_histo->histograms;
  int i, j;

  for (i = 0, j = 0; i < image_histo->size; ++i) {
    if (histograms[i] != NULL && histograms[i]->bit_cost_ != 0.) {
      if (j < i) {
        histograms[j] = histograms[i];
        histograms[i] = NULL;
      }
      ++j;
    }
  }
  image_histo->size = j;
}

static VP8LHistogram* HistogramCombineEntropyBin(
    VP8LHistogramSet* const image_histo,
    VP8LHistogram* cur_combo,
    int16_t* const bin_map, int bin_depth, int num_bins,
    double combine_cost_factor, int low_effort) {
  int bin_id;
  VP8LHistogram** const histograms = image_histo->histograms;

  for (bin_id = 0; bin_id < num_bins; ++bin_id) {
    const int bin_offset = bin_id * bin_depth;
    const int num_histos = bin_map[bin_offset];
    const int idx1 = bin_map[bin_offset + 1];
    int num_combine_failures = 0;
    int n;
    for (n = 2; n <= num_histos; ++n) {
      const int idx2 = bin_map[bin_offset + n];
      if (low_effort) {
        // Merge all histograms with the same bin index, irrespective of cost of
        // the merged histograms.
        VP8LHistogramAdd(histograms[idx1], histograms[idx2], histograms[idx1]);
        histograms[idx2]->bit_cost_ = 0.;
      } else {
        const double bit_cost_idx2 = histograms[idx2]->bit_cost_;
        if (bit_cost_idx2 > 0.) {
          const double bit_cost_thresh = -bit_cost_idx2 * combine_cost_factor;
          const double curr_cost_diff =
              HistogramAddEval(histograms[idx1], histograms[idx2],
                               cur_combo, bit_cost_thresh);
          if (curr_cost_diff < bit_cost_thresh) {
            // Try to merge two histograms only if the combo is a trivial one or
            // the two candidate histograms are already non-trivial.
            // For some images, 'try_combine' turns out to be false for a lot of
            // histogram pairs. In that case, we fallback to combining
            // histograms as usual to avoid increasing the header size.
            const int try_combine =
                (cur_combo->trivial_symbol_ != VP8L_NON_TRIVIAL_SYM) ||
                ((histograms[idx1]->trivial_symbol_ == VP8L_NON_TRIVIAL_SYM) &&
                 (histograms[idx2]->trivial_symbol_ == VP8L_NON_TRIVIAL_SYM));
            const int max_combine_failures = 32;
            if (try_combine || (num_combine_failures >= max_combine_failures)) {
              HistogramSwap(&cur_combo, &histograms[idx1]);
              histograms[idx2]->bit_cost_ = 0.;
            } else {
              ++num_combine_failures;
            }
          }
        }
      }
    }
    if (low_effort) {
      // Update the bit_cost for the merged histograms (per bin index).
      UpdateHistogramCost(histograms[idx1]);
    }
  }
  HistogramCompactBins(image_histo);
  return cur_combo;
}

static uint32_t MyRand(uint32_t *seed) {
  *seed *= 16807U;
  if (*seed == 0) {
    *seed = 1;
  }
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
  HistogramPair* heap;
  int* positions;
  int size;
  int max_index;
} HistoHeap;

static int HistoHeapInit(HistoHeap* const histo_heap, const int max_index) {
  histo_heap->size = 0;
  histo_heap->max_index = max_index;
  histo_heap->heap = WebPSafeMalloc(max_index * max_index,
                                    sizeof(*histo_heap->heap));
  histo_heap->positions = WebPSafeMalloc(max_index * max_index,
                                         sizeof(*histo_heap->positions));
  return histo_heap->heap != NULL && histo_heap->positions != NULL;
}

static void HistoHeapClear(HistoHeap* const histo_heap) {
  assert(histo_heap != NULL);
  WebPSafeFree(histo_heap->heap);
  WebPSafeFree(histo_heap->positions);
}

static void SwapHistogramPairs(HistogramPair *p1,
                               HistogramPair *p2) {
  const HistogramPair tmp = *p1;
  *p1 = *p2;
  *p2 = tmp;
}

// Given a valid min-heap in range [0, heap_size-1) this function places value
// heap[heap_size-1] into right location within heap and sets its position in
// positions array.
static void HeapPush(HistoHeap* const histo_heap) {
  HistogramPair* const heap = histo_heap->heap - 1;
  int* const positions = histo_heap->positions;
  const int max_index = histo_heap->max_index;
  int v;
  ++histo_heap->size;
  v = histo_heap->size;
  while (v > 1 && heap[v].cost_diff < heap[v >> 1].cost_diff) {
    SwapHistogramPairs(&heap[v], &heap[v >> 1]);
    // Change position of moved pair in heap.
    if (heap[v].idx1 >= 0) {
      const int pos = heap[v].idx1 * max_index + heap[v].idx2;
      assert(pos >= 0 && pos < max_index * max_index);
      positions[pos] = v;
    }
    v >>= 1;
  }
  positions[heap[v].idx1 * max_index + heap[v].idx2] = v;
}

// Given a valid min-heap in range [0, heap_size) this function shortens heap
// range by one and places element with the lowest value to (heap_size-1).
static void HeapPop(HistoHeap* const histo_heap) {
  HistogramPair* const heap = histo_heap->heap - 1;
  int* const positions = histo_heap->positions;
  const int heap_size = histo_heap->size;
  const int max_index = histo_heap->max_index;
  int v = 1;
  if (heap[v].idx1 >= 0) {
    positions[heap[v].idx1 * max_index + heap[v].idx2] = -1;
  }
  SwapHistogramPairs(&heap[v], &heap[heap_size]);
  while ((v << 1) < heap_size) {
    int son = (heap[v << 1].cost_diff < heap[v].cost_diff) ? (v << 1) : v;
    if (((v << 1) + 1) < heap_size &&
        heap[(v << 1) + 1].cost_diff < heap[son].cost_diff) {
      son = (v << 1) + 1;
    }
    if (son == v) break;
    SwapHistogramPairs(&heap[v], &heap[son]);
    // Change position of moved pair in heap.
    if (heap[v].idx1 >= 0) {
      positions[heap[v].idx1 * max_index + heap[v].idx2] = v;
    }
    v = son;
  }
  if (heap[v].idx1 >= 0) {
    positions[heap[v].idx1 * max_index + heap[v].idx2] = v;
  }
  --histo_heap->size;
}

// -----------------------------------------------------------------------------

static void PreparePair(VP8LHistogram** histograms, int idx1, int idx2,
                        HistogramPair* const pair,
                        VP8LHistogram* const histos) {
  if (idx1 > idx2) {
    const int tmp = idx2;
    idx2 = idx1;
    idx1 = tmp;
  }
  pair->idx1 = idx1;
  pair->idx2 = idx2;
  pair->cost_diff =
      HistogramAddEval(histograms[idx1], histograms[idx2], histos, 0);
  pair->cost_combo = histos->bit_cost_;
}

#define POSITION_INVALID (-1)

// Invalidates pairs intersecting (idx1, idx2) in heap.
static void InvalidatePairs(int idx1, int idx2,
                            const HistoHeap* const histo_heap) {
  HistogramPair* const heap = histo_heap->heap - 1;
  int* const positions = histo_heap->positions;
  const int max_index = histo_heap->max_index;
  int i;
  for (i = 0; i < idx1; ++i) {
    const int pos = positions[i * max_index + idx1];
    if (pos >= 0) {
      heap[pos].idx1 = POSITION_INVALID;
    }
  }
  for (i = idx1 + 1; i < max_index; ++i) {
    const int pos = positions[idx1 * max_index + i];
    if (pos >= 0) {
      heap[pos].idx1 = POSITION_INVALID;
    }
  }
  for (i = 0; i < idx2; ++i) {
    const int pos = positions[i * max_index + idx2];
    if (pos >= 0) {
      heap[pos].idx1 = POSITION_INVALID;
    }
  }
  for (i = idx2 + 1; i < max_index; ++i) {
    const int pos = positions[idx2 * max_index + i];
    if (pos >= 0) {
      heap[pos].idx1 = POSITION_INVALID;
    }
  }
}

// Combines histograms by continuously choosing the one with the highest cost
// reduction.
static int HistogramCombineGreedy(VP8LHistogramSet* const image_histo,
                                  VP8LHistogram* const histos) {
  int ok = 0;
  int image_histo_size = image_histo->size;
  int i, j;
  VP8LHistogram** const histograms = image_histo->histograms;
  // Indexes of remaining histograms.
  int* const clusters = WebPSafeMalloc(image_histo_size, sizeof(*clusters));
  // Heap of histogram pairs.
  HistoHeap histo_heap;

  if (!HistoHeapInit(&histo_heap, image_histo_size) || clusters == NULL) {
    goto End;
  }

  for (i = 0; i < image_histo_size; ++i) {
    // Initialize clusters indexes.
    clusters[i] = i;
    for (j = i + 1; j < image_histo_size; ++j) {
      // Initialize positions array.
      histo_heap.positions[i * histo_heap.max_index + j] = POSITION_INVALID;
      PreparePair(histograms, i, j, &histo_heap.heap[histo_heap.size], histos);
      if (histo_heap.heap[histo_heap.size].cost_diff < 0) {
        HeapPush(&histo_heap);
      }
    }
  }

  while (image_histo_size > 1 && histo_heap.size > 0) {
    const int idx1 = histo_heap.heap[0].idx1;
    const int idx2 = histo_heap.heap[0].idx2;
    VP8LHistogramAdd(histograms[idx2], histograms[idx1], histograms[idx1]);
    histograms[idx1]->bit_cost_ = histo_heap.heap[0].cost_combo;
    // Remove merged histogram.
    for (i = 0; i + 1 < image_histo_size; ++i) {
      if (clusters[i] >= idx2) {
        clusters[i] = clusters[i + 1];
      }
    }
    --image_histo_size;

    // Invalidate pairs intersecting the just combined best pair.
    InvalidatePairs(idx1, idx2, &histo_heap);

    // Pop invalid pairs from the top of the heap.
    while (histo_heap.size > 0 && histo_heap.heap[0].idx1 < 0) {
      HeapPop(&histo_heap);
    }

    // Push new pairs formed with combined histogram to the heap.
    for (i = 0; i < image_histo_size; ++i) {
      if (clusters[i] != idx1) {
        PreparePair(histograms, idx1, clusters[i],
                    &histo_heap.heap[histo_heap.size], histos);
        if (histo_heap.heap[histo_heap.size].cost_diff < 0) {
          HeapPush(&histo_heap);
        }
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
  HistoHeapClear(&histo_heap);
  return ok;
}

static VP8LHistogram* HistogramCombineStochastic(
    VP8LHistogramSet* const image_histo,
    VP8LHistogram* tmp_histo,
    VP8LHistogram* best_combo,
    int quality, int min_cluster_size) {
  int iter;
  uint32_t seed = 0;
  int tries_with_no_success = 0;
  int image_histo_size = image_histo->size;
  const int iter_mult = (quality < 25) ? 2 : 2 + (quality - 25) / 8;
  const int outer_iters = image_histo_size * iter_mult;
  const int num_pairs = image_histo_size / 2;
  const int num_tries_no_success = outer_iters / 2;
  VP8LHistogram** const histograms = image_histo->histograms;

  // Collapse similar histograms in 'image_histo'.
  ++min_cluster_size;
  for (iter = 0;
       iter < outer_iters && image_histo_size >= min_cluster_size;
       ++iter) {
    double best_cost_diff = 0.;
    int best_idx1 = -1, best_idx2 = 1;
    int j;
    const int num_tries =
        (num_pairs < image_histo_size) ? num_pairs : image_histo_size;
    seed += iter;
    for (j = 0; j < num_tries; ++j) {
      double curr_cost_diff;
      // Choose two histograms at random and try to combine them.
      const uint32_t idx1 = MyRand(&seed) % image_histo_size;
      const uint32_t tmp = (j & 7) + 1;
      const uint32_t diff =
          (tmp < 3) ? tmp : MyRand(&seed) % (image_histo_size - 1);
      const uint32_t idx2 = (idx1 + diff + 1) % image_histo_size;
      if (idx1 == idx2) {
        continue;
      }

      // Calculate cost reduction on combining.
      curr_cost_diff = HistogramAddEval(histograms[idx1], histograms[idx2],
                                        tmp_histo, best_cost_diff);
      if (curr_cost_diff < best_cost_diff) {    // found a better pair?
        HistogramSwap(&best_combo, &tmp_histo);
        best_cost_diff = curr_cost_diff;
        best_idx1 = idx1;
        best_idx2 = idx2;
      }
    }

    if (best_idx1 >= 0) {
      HistogramSwap(&best_combo, &histograms[best_idx1]);
      // swap best_idx2 slot with last one (which is now unused)
      --image_histo_size;
      if (best_idx2 != image_histo_size) {
        HistogramSwap(&histograms[image_histo_size], &histograms[best_idx2]);
        histograms[image_histo_size] = NULL;
      }
      tries_with_no_success = 0;
    }
    if (++tries_with_no_success >= num_tries_no_success) {
      break;
    }
  }
  image_histo->size = image_histo_size;
  return best_combo;
}

// -----------------------------------------------------------------------------
// Histogram refinement

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
static void HistogramRemap(const VP8LHistogramSet* const orig_histo,
                           const VP8LHistogramSet* const image_histo,
                           uint16_t* const symbols) {
  int i;
  VP8LHistogram** const orig_histograms = orig_histo->histograms;
  VP8LHistogram** const histograms = image_histo->histograms;
  const int orig_histo_size = orig_histo->size;
  const int image_histo_size = image_histo->size;
  if (image_histo_size > 1) {
    for (i = 0; i < orig_histo_size; ++i) {
      int best_out = 0;
      double best_bits =
          HistogramAddThresh(histograms[0], orig_histograms[i], MAX_COST);
      int k;
      for (k = 1; k < image_histo_size; ++k) {
        const double cur_bits =
            HistogramAddThresh(histograms[k], orig_histograms[i], best_bits);
        if (cur_bits < best_bits) {
          best_bits = cur_bits;
          best_out = k;
        }
      }
      symbols[i] = best_out;
    }
  } else {
    assert(image_histo_size == 1);
    for (i = 0; i < orig_histo_size; ++i) {
      symbols[i] = 0;
    }
  }

  // Recompute each out based on raw and symbols.
  for (i = 0; i < image_histo_size; ++i) {
    HistogramClear(histograms[i]);
  }

  for (i = 0; i < orig_histo_size; ++i) {
    const int idx = symbols[i];
    VP8LHistogramAdd(orig_histograms[i], histograms[idx], histograms[idx]);
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
                             VP8LHistogramSet* const tmp_histos,
                             uint16_t* const histogram_symbols) {
  int ok = 0;
  const int histo_xsize = histo_bits ? VP8LSubSampleSize(xsize, histo_bits) : 1;
  const int histo_ysize = histo_bits ? VP8LSubSampleSize(ysize, histo_bits) : 1;
  const int image_histo_raw_size = histo_xsize * histo_ysize;
  const int entropy_combine_num_bins = low_effort ? NUM_PARTITIONS : BIN_SIZE;

  // The bin_map for every bin follows following semantics:
  // bin_map[n][0] = num_histo; // The number of histograms in that bin.
  // bin_map[n][1] = index of first histogram in that bin;
  // bin_map[n][num_histo] = index of last histogram in that bin;
  // bin_map[n][num_histo + 1] ... bin_map[n][bin_depth - 1] = unused indices.
  const int bin_depth = image_histo_raw_size + 1;
  int16_t* bin_map = NULL;
  VP8LHistogramSet* const orig_histo =
      VP8LAllocateHistogramSet(image_histo_raw_size, cache_bits);
  VP8LHistogram* cur_combo;
  const int entropy_combine =
      (orig_histo->size > entropy_combine_num_bins * 2) && (quality < 100);

  if (orig_histo == NULL) goto Error;

  // Don't attempt linear bin-partition heuristic for:
  // histograms of small sizes, as bin_map will be very sparse and;
  // Maximum quality (q==100), to preserve the compression gains at that level.
  if (entropy_combine) {
    const int bin_map_size = bin_depth * entropy_combine_num_bins;
    bin_map = (int16_t*)WebPSafeCalloc(bin_map_size, sizeof(*bin_map));
    if (bin_map == NULL) goto Error;
  }

  // Construct the histograms from backward references.
  HistogramBuild(xsize, histo_bits, refs, orig_histo);
  // Copies the histograms and computes its bit_cost.
  HistogramCopyAndAnalyze(orig_histo, image_histo);

  cur_combo = tmp_histos->histograms[1];  // pick up working slot
  if (entropy_combine) {
    const double combine_cost_factor =
        GetCombineCostFactor(image_histo_raw_size, quality);
    HistogramAnalyzeEntropyBin(orig_histo, bin_map, low_effort);
    // Collapse histograms with similar entropy.
    cur_combo = HistogramCombineEntropyBin(image_histo, cur_combo, bin_map,
                                           bin_depth, entropy_combine_num_bins,
                                           combine_cost_factor, low_effort);
  }

  // Don't combine the histograms using stochastic and greedy heuristics for
  // low-effort compression mode.
  if (!low_effort || !entropy_combine) {
    const float x = quality / 100.f;
    // cubic ramp between 1 and MAX_HISTO_GREEDY:
    const int threshold_size = (int)(1 + (x * x * x) * (MAX_HISTO_GREEDY - 1));
    cur_combo = HistogramCombineStochastic(image_histo,
                                           tmp_histos->histograms[0],
                                           cur_combo, quality, threshold_size);
    if ((image_histo->size <= threshold_size) &&
        !HistogramCombineGreedy(image_histo, cur_combo)) {
      goto Error;
    }
  }

  // TODO(vikasa): Optimize HistogramRemap for low-effort compression mode also.
  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(orig_histo, image_histo, histogram_symbols);

  ok = 1;

 Error:
  WebPSafeFree(bin_map);
  VP8LFreeHistogramSet(orig_histo);
  return ok;
}
