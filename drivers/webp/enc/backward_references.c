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

#include <assert.h>
#include <math.h>

#include "./backward_references.h"
#include "./histogram.h"
#include "../dsp/lossless.h"
#include "../utils/color_cache.h"
#include "../utils/utils.h"

#define VALUES_IN_BYTE 256

#define MIN_BLOCK_SIZE 256  // minimum block size for backward references

#define MAX_ENTROPY    (1e30f)

// 1M window (4M bytes) minus 120 special codes for short distances.
#define WINDOW_SIZE ((1 << 20) - 120)

// Bounds for the match length.
#define MIN_LENGTH 2
#define MAX_LENGTH 4096

// -----------------------------------------------------------------------------

static const uint8_t plane_to_code_lut[128] = {
 96,   73,  55,  39,  23,  13,   5,  1,  255, 255, 255, 255, 255, 255, 255, 255,
 101,  78,  58,  42,  26,  16,   8,  2,    0,   3,  9,   17,  27,  43,  59,  79,
 102,  86,  62,  46,  32,  20,  10,  6,    4,   7,  11,  21,  33,  47,  63,  87,
 105,  90,  70,  52,  37,  28,  18,  14,  12,  15,  19,  29,  38,  53,  71,  91,
 110,  99,  82,  66,  48,  35,  30,  24,  22,  25,  31,  36,  49,  67,  83, 100,
 115, 108,  94,  76,  64,  50,  44,  40,  34,  41,  45,  51,  65,  77,  95, 109,
 118, 113, 103,  92,  80,  68,  60,  56,  54,  57,  61,  69,  81,  93, 104, 114,
 119, 116, 111, 106,  97,  88,  84,  74,  72,  75,  85,  89,  98, 107, 112, 117
};

static int DistanceToPlaneCode(int xsize, int dist) {
  const int yoffset = dist / xsize;
  const int xoffset = dist - yoffset * xsize;
  if (xoffset <= 8 && yoffset < 8) {
    return plane_to_code_lut[yoffset * 16 + 8 - xoffset] + 1;
  } else if (xoffset > xsize - 8 && yoffset < 7) {
    return plane_to_code_lut[(yoffset + 1) * 16 + 8 + (xsize - xoffset)] + 1;
  }
  return dist + 120;
}

static WEBP_INLINE int FindMatchLength(const uint32_t* const array1,
                                       const uint32_t* const array2,
                                       int best_len_match,
                                       int max_limit) {
#if !defined(__x86_64__)
  // TODO(vrabaud): Compare on other architectures.
  int match_len = 0;
  // Before 'expensive' linear match, check if the two arrays match at the
  // current best length index.
  if (array1[best_len_match] != array2[best_len_match]) return 0;
  while (match_len < max_limit && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
#else
  const uint32_t* array1_32 = array1;
  const uint32_t* array2_32 = array2;
  // max value is aligned to (uint64_t*) array1
  const uint32_t* const array1_32_max = array1 + (max_limit & ~1);

  // Before 'expensive' linear match, check if the two arrays match at the
  // current best length index.
  if (array1[best_len_match] != array2[best_len_match]) return 0;

  // TODO(vrabaud): add __predict_true on bound checking?
  while (array1_32 < array1_32_max) {
    if (*(uint64_t*)array1_32 == *(uint64_t*)array2_32) {
      array1_32 += 2;
      array2_32 += 2;
    } else {
      // if the uint32_t pointed to are the same, then the following ones have
      // to be different
      return (array1_32 - array1) + (*array1_32 == *array2_32);
    }
  }

  // Deal with the potential last uint32_t.
  if ((max_limit & 1) && (*array1_32 != *array2_32)) return max_limit - 1;
  return max_limit;
#endif
}

// -----------------------------------------------------------------------------
//  VP8LBackwardRefs

struct PixOrCopyBlock {
  PixOrCopyBlock* next_;   // next block (or NULL)
  PixOrCopy* start_;       // data start
  int size_;               // currently used size
};

static void ClearBackwardRefs(VP8LBackwardRefs* const refs) {
  assert(refs != NULL);
  if (refs->tail_ != NULL) {
    *refs->tail_ = refs->free_blocks_;  // recycle all blocks at once
  }
  refs->free_blocks_ = refs->refs_;
  refs->tail_ = &refs->refs_;
  refs->last_block_ = NULL;
  refs->refs_ = NULL;
}

void VP8LBackwardRefsClear(VP8LBackwardRefs* const refs) {
  assert(refs != NULL);
  ClearBackwardRefs(refs);
  while (refs->free_blocks_ != NULL) {
    PixOrCopyBlock* const next = refs->free_blocks_->next_;
    WebPSafeFree(refs->free_blocks_);
    refs->free_blocks_ = next;
  }
}

void VP8LBackwardRefsInit(VP8LBackwardRefs* const refs, int block_size) {
  assert(refs != NULL);
  memset(refs, 0, sizeof(*refs));
  refs->tail_ = &refs->refs_;
  refs->block_size_ =
      (block_size < MIN_BLOCK_SIZE) ? MIN_BLOCK_SIZE : block_size;
}

VP8LRefsCursor VP8LRefsCursorInit(const VP8LBackwardRefs* const refs) {
  VP8LRefsCursor c;
  c.cur_block_ = refs->refs_;
  if (refs->refs_ != NULL) {
    c.cur_pos = c.cur_block_->start_;
    c.last_pos_ = c.cur_pos + c.cur_block_->size_;
  } else {
    c.cur_pos = NULL;
    c.last_pos_ = NULL;
  }
  return c;
}

void VP8LRefsCursorNextBlock(VP8LRefsCursor* const c) {
  PixOrCopyBlock* const b = c->cur_block_->next_;
  c->cur_pos = (b == NULL) ? NULL : b->start_;
  c->last_pos_ = (b == NULL) ? NULL : b->start_ + b->size_;
  c->cur_block_ = b;
}

// Create a new block, either from the free list or allocated
static PixOrCopyBlock* BackwardRefsNewBlock(VP8LBackwardRefs* const refs) {
  PixOrCopyBlock* b = refs->free_blocks_;
  if (b == NULL) {   // allocate new memory chunk
    const size_t total_size =
        sizeof(*b) + refs->block_size_ * sizeof(*b->start_);
    b = (PixOrCopyBlock*)WebPSafeMalloc(1ULL, total_size);
    if (b == NULL) {
      refs->error_ |= 1;
      return NULL;
    }
    b->start_ = (PixOrCopy*)((uint8_t*)b + sizeof(*b));  // not always aligned
  } else {  // recycle from free-list
    refs->free_blocks_ = b->next_;
  }
  *refs->tail_ = b;
  refs->tail_ = &b->next_;
  refs->last_block_ = b;
  b->next_ = NULL;
  b->size_ = 0;
  return b;
}

static WEBP_INLINE void BackwardRefsCursorAdd(VP8LBackwardRefs* const refs,
                                              const PixOrCopy v) {
  PixOrCopyBlock* b = refs->last_block_;
  if (b == NULL || b->size_ == refs->block_size_) {
    b = BackwardRefsNewBlock(refs);
    if (b == NULL) return;   // refs->error_ is set
  }
  b->start_[b->size_++] = v;
}

int VP8LBackwardRefsCopy(const VP8LBackwardRefs* const src,
                         VP8LBackwardRefs* const dst) {
  const PixOrCopyBlock* b = src->refs_;
  ClearBackwardRefs(dst);
  assert(src->block_size_ == dst->block_size_);
  while (b != NULL) {
    PixOrCopyBlock* const new_b = BackwardRefsNewBlock(dst);
    if (new_b == NULL) return 0;   // dst->error_ is set
    memcpy(new_b->start_, b->start_, b->size_ * sizeof(*b->start_));
    new_b->size_ = b->size_;
    b = b->next_;
  }
  return 1;
}

// -----------------------------------------------------------------------------
// Hash chains

// initialize as empty
static void HashChainReset(VP8LHashChain* const p) {
  int i;
  assert(p != NULL);
  for (i = 0; i < p->size_; ++i) {
    p->chain_[i] = -1;
  }
  for (i = 0; i < HASH_SIZE; ++i) {
    p->hash_to_first_index_[i] = -1;
  }
}

int VP8LHashChainInit(VP8LHashChain* const p, int size) {
  assert(p->size_ == 0);
  assert(p->chain_ == NULL);
  assert(size > 0);
  p->chain_ = (int*)WebPSafeMalloc(size, sizeof(*p->chain_));
  if (p->chain_ == NULL) return 0;
  p->size_ = size;
  HashChainReset(p);
  return 1;
}

void VP8LHashChainClear(VP8LHashChain* const p) {
  assert(p != NULL);
  WebPSafeFree(p->chain_);
  p->size_ = 0;
  p->chain_ = NULL;
}

// -----------------------------------------------------------------------------

#define HASH_MULTIPLIER_HI (0xc6a4a793U)
#define HASH_MULTIPLIER_LO (0x5bd1e996U)

static WEBP_INLINE uint32_t GetPixPairHash64(const uint32_t* const argb) {
  uint32_t key;
  key  = argb[1] * HASH_MULTIPLIER_HI;
  key += argb[0] * HASH_MULTIPLIER_LO;
  key = key >> (32 - HASH_BITS);
  return key;
}

// Insertion of two pixels at a time.
static void HashChainInsert(VP8LHashChain* const p,
                            const uint32_t* const argb, int pos) {
  const uint32_t hash_code = GetPixPairHash64(argb);
  p->chain_[pos] = p->hash_to_first_index_[hash_code];
  p->hash_to_first_index_[hash_code] = pos;
}

// Returns the maximum number of hash chain lookups to do for a
// given compression quality. Return value in range [6, 86].
static int GetMaxItersForQuality(int quality, int low_effort) {
  return (low_effort ? 6 : 8) + (quality * quality) / 128;
}

static int GetWindowSizeForHashChain(int quality, int xsize) {
  const int max_window_size = (quality > 75) ? WINDOW_SIZE
                            : (quality > 50) ? (xsize << 8)
                            : (quality > 25) ? (xsize << 6)
                            : (xsize << 4);
  assert(xsize > 0);
  return (max_window_size > WINDOW_SIZE) ? WINDOW_SIZE : max_window_size;
}

static WEBP_INLINE int MaxFindCopyLength(int len) {
  return (len < MAX_LENGTH) ? len : MAX_LENGTH;
}

static void HashChainFindOffset(const VP8LHashChain* const p, int base_position,
                                const uint32_t* const argb, int len,
                                int window_size, int* const distance_ptr) {
  const uint32_t* const argb_start = argb + base_position;
  const int min_pos =
      (base_position > window_size) ? base_position - window_size : 0;
  int pos;
  assert(len <= MAX_LENGTH);
  for (pos = p->hash_to_first_index_[GetPixPairHash64(argb_start)];
       pos >= min_pos;
       pos = p->chain_[pos]) {
    const int curr_length =
        FindMatchLength(argb + pos, argb_start, len - 1, len);
    if (curr_length == len) break;
  }
  *distance_ptr = base_position - pos;
}

static int HashChainFindCopy(const VP8LHashChain* const p,
                             int base_position,
                             const uint32_t* const argb, int max_len,
                             int window_size, int iter_max,
                             int* const distance_ptr,
                             int* const length_ptr) {
  const uint32_t* const argb_start = argb + base_position;
  int iter = iter_max;
  int best_length = 0;
  int best_distance = 0;
  const int min_pos =
      (base_position > window_size) ? base_position - window_size : 0;
  int pos;
  int length_max = 256;
  if (max_len < length_max) {
    length_max = max_len;
  }
  for (pos = p->hash_to_first_index_[GetPixPairHash64(argb_start)];
       pos >= min_pos;
       pos = p->chain_[pos]) {
    int curr_length;
    int distance;
    if (--iter < 0) {
      break;
    }

    curr_length = FindMatchLength(argb + pos, argb_start, best_length, max_len);
    if (best_length < curr_length) {
      distance = base_position - pos;
      best_length = curr_length;
      best_distance = distance;
      if (curr_length >= length_max) {
        break;
      }
    }
  }
  *distance_ptr = best_distance;
  *length_ptr = best_length;
  return (best_length >= MIN_LENGTH);
}

static WEBP_INLINE void AddSingleLiteral(uint32_t pixel, int use_color_cache,
                                         VP8LColorCache* const hashers,
                                         VP8LBackwardRefs* const refs) {
  PixOrCopy v;
  if (use_color_cache) {
    const uint32_t key = VP8LColorCacheGetIndex(hashers, pixel);
    if (VP8LColorCacheLookup(hashers, key) == pixel) {
      v = PixOrCopyCreateCacheIdx(key);
    } else {
      v = PixOrCopyCreateLiteral(pixel);
      VP8LColorCacheSet(hashers, key, pixel);
    }
  } else {
    v = PixOrCopyCreateLiteral(pixel);
  }
  BackwardRefsCursorAdd(refs, v);
}

static int BackwardReferencesRle(int xsize, int ysize,
                                 const uint32_t* const argb,
                                 int cache_bits, VP8LBackwardRefs* const refs) {
  const int pix_count = xsize * ysize;
  int i, k;
  const int use_color_cache = (cache_bits > 0);
  VP8LColorCache hashers;

  if (use_color_cache && !VP8LColorCacheInit(&hashers, cache_bits)) {
    return 0;
  }
  ClearBackwardRefs(refs);
  // Add first pixel as literal.
  AddSingleLiteral(argb[0], use_color_cache, &hashers, refs);
  i = 1;
  while (i < pix_count) {
    const int max_len = MaxFindCopyLength(pix_count - i);
    const int kMinLength = 4;
    const int rle_len = FindMatchLength(argb + i, argb + i - 1, 0, max_len);
    const int prev_row_len = (i < xsize) ? 0 :
        FindMatchLength(argb + i, argb + i - xsize, 0, max_len);
    if (rle_len >= prev_row_len && rle_len >= kMinLength) {
      BackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(1, rle_len));
      // We don't need to update the color cache here since it is always the
      // same pixel being copied, and that does not change the color cache
      // state.
      i += rle_len;
    } else if (prev_row_len >= kMinLength) {
      BackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(xsize, prev_row_len));
      if (use_color_cache) {
        for (k = 0; k < prev_row_len; ++k) {
          VP8LColorCacheInsert(&hashers, argb[i + k]);
        }
      }
      i += prev_row_len;
    } else {
      AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
      i++;
    }
  }
  if (use_color_cache) VP8LColorCacheClear(&hashers);
  return !refs->error_;
}

static int BackwardReferencesLz77(int xsize, int ysize,
                                  const uint32_t* const argb, int cache_bits,
                                  int quality, int low_effort,
                                  VP8LHashChain* const hash_chain,
                                  VP8LBackwardRefs* const refs) {
  int i;
  int ok = 0;
  int cc_init = 0;
  const int use_color_cache = (cache_bits > 0);
  const int pix_count = xsize * ysize;
  VP8LColorCache hashers;
  int iter_max = GetMaxItersForQuality(quality, low_effort);
  const int window_size = GetWindowSizeForHashChain(quality, xsize);
  int min_matches = 32;

  if (use_color_cache) {
    cc_init = VP8LColorCacheInit(&hashers, cache_bits);
    if (!cc_init) goto Error;
  }
  ClearBackwardRefs(refs);
  HashChainReset(hash_chain);
  for (i = 0; i < pix_count - 2; ) {
    // Alternative#1: Code the pixels starting at 'i' using backward reference.
    int offset = 0;
    int len = 0;
    const int max_len = MaxFindCopyLength(pix_count - i);
    HashChainFindCopy(hash_chain, i, argb, max_len, window_size,
                      iter_max, &offset, &len);
    if (len > MIN_LENGTH || (len == MIN_LENGTH && offset <= 512)) {
      int offset2 = 0;
      int len2 = 0;
      int k;
      min_matches = 8;
      HashChainInsert(hash_chain, &argb[i], i);
      if ((len < (max_len >> 2)) && !low_effort) {
        // Evaluate Alternative#2: Insert the pixel at 'i' as literal, and code
        // the pixels starting at 'i + 1' using backward reference.
        HashChainFindCopy(hash_chain, i + 1, argb, max_len - 1,
                          window_size, iter_max, &offset2,
                          &len2);
        if (len2 > len + 1) {
          AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
          i++;  // Backward reference to be done for next pixel.
          len = len2;
          offset = offset2;
        }
      }
      BackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(offset, len));
      if (use_color_cache) {
        for (k = 0; k < len; ++k) {
          VP8LColorCacheInsert(&hashers, argb[i + k]);
        }
      }
      // Add to the hash_chain (but cannot add the last pixel).
      if (offset >= 3 && offset != xsize) {
        const int last = (len < pix_count - 1 - i) ? len : pix_count - 1 - i;
        for (k = 2; k < last - 8; k += 2) {
          HashChainInsert(hash_chain, &argb[i + k], i + k);
        }
        for (; k < last; ++k) {
          HashChainInsert(hash_chain, &argb[i + k], i + k);
        }
      }
      i += len;
    } else {
      AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
      HashChainInsert(hash_chain, &argb[i], i);
      ++i;
      --min_matches;
      if (min_matches <= 0) {
        AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
        HashChainInsert(hash_chain, &argb[i], i);
        ++i;
      }
    }
  }
  while (i < pix_count) {
    // Handle the last pixel(s).
    AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
    ++i;
  }

  ok = !refs->error_;
 Error:
  if (cc_init) VP8LColorCacheClear(&hashers);
  return ok;
}

// -----------------------------------------------------------------------------

typedef struct {
  double alpha_[VALUES_IN_BYTE];
  double red_[VALUES_IN_BYTE];
  double blue_[VALUES_IN_BYTE];
  double distance_[NUM_DISTANCE_CODES];
  double* literal_;
} CostModel;

static int BackwardReferencesTraceBackwards(
    int xsize, int ysize, const uint32_t* const argb, int quality,
    int cache_bits, VP8LHashChain* const hash_chain,
    VP8LBackwardRefs* const refs);

static void ConvertPopulationCountTableToBitEstimates(
    int num_symbols, const uint32_t population_counts[], double output[]) {
  uint32_t sum = 0;
  int nonzeros = 0;
  int i;
  for (i = 0; i < num_symbols; ++i) {
    sum += population_counts[i];
    if (population_counts[i] > 0) {
      ++nonzeros;
    }
  }
  if (nonzeros <= 1) {
    memset(output, 0, num_symbols * sizeof(*output));
  } else {
    const double logsum = VP8LFastLog2(sum);
    for (i = 0; i < num_symbols; ++i) {
      output[i] = logsum - VP8LFastLog2(population_counts[i]);
    }
  }
}

static int CostModelBuild(CostModel* const m, int cache_bits,
                          VP8LBackwardRefs* const refs) {
  int ok = 0;
  VP8LHistogram* const histo = VP8LAllocateHistogram(cache_bits);
  if (histo == NULL) goto Error;

  VP8LHistogramCreate(histo, refs, cache_bits);

  ConvertPopulationCountTableToBitEstimates(
      VP8LHistogramNumCodes(histo->palette_code_bits_),
      histo->literal_, m->literal_);
  ConvertPopulationCountTableToBitEstimates(
      VALUES_IN_BYTE, histo->red_, m->red_);
  ConvertPopulationCountTableToBitEstimates(
      VALUES_IN_BYTE, histo->blue_, m->blue_);
  ConvertPopulationCountTableToBitEstimates(
      VALUES_IN_BYTE, histo->alpha_, m->alpha_);
  ConvertPopulationCountTableToBitEstimates(
      NUM_DISTANCE_CODES, histo->distance_, m->distance_);
  ok = 1;

 Error:
  VP8LFreeHistogram(histo);
  return ok;
}

static WEBP_INLINE double GetLiteralCost(const CostModel* const m, uint32_t v) {
  return m->alpha_[v >> 24] +
         m->red_[(v >> 16) & 0xff] +
         m->literal_[(v >> 8) & 0xff] +
         m->blue_[v & 0xff];
}

static WEBP_INLINE double GetCacheCost(const CostModel* const m, uint32_t idx) {
  const int literal_idx = VALUES_IN_BYTE + NUM_LENGTH_CODES + idx;
  return m->literal_[literal_idx];
}

static WEBP_INLINE double GetLengthCost(const CostModel* const m,
                                        uint32_t length) {
  int code, extra_bits;
  VP8LPrefixEncodeBits(length, &code, &extra_bits);
  return m->literal_[VALUES_IN_BYTE + code] + extra_bits;
}

static WEBP_INLINE double GetDistanceCost(const CostModel* const m,
                                          uint32_t distance) {
  int code, extra_bits;
  VP8LPrefixEncodeBits(distance, &code, &extra_bits);
  return m->distance_[code] + extra_bits;
}

static void AddSingleLiteralWithCostModel(
    const uint32_t* const argb, VP8LHashChain* const hash_chain,
    VP8LColorCache* const hashers, const CostModel* const cost_model, int idx,
    int is_last, int use_color_cache, double prev_cost, float* const cost,
    uint16_t* const dist_array) {
  double cost_val = prev_cost;
  const uint32_t color = argb[0];
  if (!is_last) {
    HashChainInsert(hash_chain, argb, idx);
  }
  if (use_color_cache && VP8LColorCacheContains(hashers, color)) {
    const double mul0 = 0.68;
    const int ix = VP8LColorCacheGetIndex(hashers, color);
    cost_val += GetCacheCost(cost_model, ix) * mul0;
  } else {
    const double mul1 = 0.82;
    if (use_color_cache) VP8LColorCacheInsert(hashers, color);
    cost_val += GetLiteralCost(cost_model, color) * mul1;
  }
  if (cost[idx] > cost_val) {
    cost[idx] = (float)cost_val;
    dist_array[idx] = 1;  // only one is inserted.
  }
}

static int BackwardReferencesHashChainDistanceOnly(
    int xsize, int ysize, const uint32_t* const argb,
    int quality, int cache_bits, VP8LHashChain* const hash_chain,
    VP8LBackwardRefs* const refs, uint16_t* const dist_array) {
  int i;
  int ok = 0;
  int cc_init = 0;
  const int pix_count = xsize * ysize;
  const int use_color_cache = (cache_bits > 0);
  float* const cost =
      (float*)WebPSafeMalloc(pix_count, sizeof(*cost));
  const size_t literal_array_size = sizeof(double) *
      (NUM_LITERAL_CODES + NUM_LENGTH_CODES +
       ((cache_bits > 0) ? (1 << cache_bits) : 0));
  const size_t cost_model_size = sizeof(CostModel) + literal_array_size;
  CostModel* const cost_model =
      (CostModel*)WebPSafeMalloc(1ULL, cost_model_size);
  VP8LColorCache hashers;
  const int skip_length = 32 + quality;
  const int skip_min_distance_code = 2;
  int iter_max = GetMaxItersForQuality(quality, 0);
  const int window_size = GetWindowSizeForHashChain(quality, xsize);

  if (cost == NULL || cost_model == NULL) goto Error;

  cost_model->literal_ = (double*)(cost_model + 1);
  if (use_color_cache) {
    cc_init = VP8LColorCacheInit(&hashers, cache_bits);
    if (!cc_init) goto Error;
  }

  if (!CostModelBuild(cost_model, cache_bits, refs)) {
    goto Error;
  }

  for (i = 0; i < pix_count; ++i) cost[i] = 1e38f;

  // We loop one pixel at a time, but store all currently best points to
  // non-processed locations from this point.
  dist_array[0] = 0;
  HashChainReset(hash_chain);
  // Add first pixel as literal.
  AddSingleLiteralWithCostModel(argb + 0, hash_chain, &hashers, cost_model, 0,
                                0, use_color_cache, 0.0, cost, dist_array);
  for (i = 1; i < pix_count - 1; ++i) {
    int offset = 0;
    int len = 0;
    double prev_cost = cost[i - 1];
    const int max_len = MaxFindCopyLength(pix_count - i);
    HashChainFindCopy(hash_chain, i, argb, max_len, window_size,
                      iter_max, &offset, &len);
    if (len >= MIN_LENGTH) {
      const int code = DistanceToPlaneCode(xsize, offset);
      const double distance_cost =
          prev_cost + GetDistanceCost(cost_model, code);
      int k;
      for (k = 1; k < len; ++k) {
        const double cost_val = distance_cost + GetLengthCost(cost_model, k);
        if (cost[i + k] > cost_val) {
          cost[i + k] = (float)cost_val;
          dist_array[i + k] = k + 1;
        }
      }
      // This if is for speedup only. It roughly doubles the speed, and
      // makes compression worse by .1 %.
      if (len >= skip_length && code <= skip_min_distance_code) {
        // Long copy for short distances, let's skip the middle
        // lookups for better copies.
        // 1) insert the hashes.
        if (use_color_cache) {
          for (k = 0; k < len; ++k) {
            VP8LColorCacheInsert(&hashers, argb[i + k]);
          }
        }
        // 2) Add to the hash_chain (but cannot add the last pixel)
        {
          const int last = (len + i < pix_count - 1) ? len + i
                                                     : pix_count - 1;
          for (k = i; k < last; ++k) {
            HashChainInsert(hash_chain, &argb[k], k);
          }
        }
        // 3) jump.
        i += len - 1;  // for loop does ++i, thus -1 here.
        goto next_symbol;
      }
      if (len != MIN_LENGTH) {
        int code_min_length;
        double cost_total;
        HashChainFindOffset(hash_chain, i, argb, MIN_LENGTH, window_size,
                            &offset);
        code_min_length = DistanceToPlaneCode(xsize, offset);
        cost_total = prev_cost +
            GetDistanceCost(cost_model, code_min_length) +
            GetLengthCost(cost_model, 1);
        if (cost[i + 1] > cost_total) {
          cost[i + 1] = (float)cost_total;
          dist_array[i + 1] = 2;
        }
      }
    }
    AddSingleLiteralWithCostModel(argb + i, hash_chain, &hashers, cost_model, i,
                                  0, use_color_cache, prev_cost, cost,
                                  dist_array);
 next_symbol: ;
  }
  // Handle the last pixel.
  if (i == (pix_count - 1)) {
    AddSingleLiteralWithCostModel(argb + i, hash_chain, &hashers, cost_model, i,
                                  1, use_color_cache, cost[pix_count - 2], cost,
                                  dist_array);
  }
  ok = !refs->error_;
 Error:
  if (cc_init) VP8LColorCacheClear(&hashers);
  WebPSafeFree(cost_model);
  WebPSafeFree(cost);
  return ok;
}

// We pack the path at the end of *dist_array and return
// a pointer to this part of the array. Example:
// dist_array = [1x2xx3x2] => packed [1x2x1232], chosen_path = [1232]
static void TraceBackwards(uint16_t* const dist_array,
                           int dist_array_size,
                           uint16_t** const chosen_path,
                           int* const chosen_path_size) {
  uint16_t* path = dist_array + dist_array_size;
  uint16_t* cur = dist_array + dist_array_size - 1;
  while (cur >= dist_array) {
    const int k = *cur;
    --path;
    *path = k;
    cur -= k;
  }
  *chosen_path = path;
  *chosen_path_size = (int)(dist_array + dist_array_size - path);
}

static int BackwardReferencesHashChainFollowChosenPath(
    int xsize, int ysize, const uint32_t* const argb,
    int quality, int cache_bits,
    const uint16_t* const chosen_path, int chosen_path_size,
    VP8LHashChain* const hash_chain,
    VP8LBackwardRefs* const refs) {
  const int pix_count = xsize * ysize;
  const int use_color_cache = (cache_bits > 0);
  int ix;
  int i = 0;
  int ok = 0;
  int cc_init = 0;
  const int window_size = GetWindowSizeForHashChain(quality, xsize);
  VP8LColorCache hashers;

  if (use_color_cache) {
    cc_init = VP8LColorCacheInit(&hashers, cache_bits);
    if (!cc_init) goto Error;
  }

  ClearBackwardRefs(refs);
  HashChainReset(hash_chain);
  for (ix = 0; ix < chosen_path_size; ++ix) {
    int offset = 0;
    const int len = chosen_path[ix];
    if (len != 1) {
      int k;
      HashChainFindOffset(hash_chain, i, argb, len, window_size, &offset);
      BackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(offset, len));
      if (use_color_cache) {
        for (k = 0; k < len; ++k) {
          VP8LColorCacheInsert(&hashers, argb[i + k]);
        }
      }
      {
        const int last = (len < pix_count - 1 - i) ? len : pix_count - 1 - i;
        for (k = 0; k < last; ++k) {
          HashChainInsert(hash_chain, &argb[i + k], i + k);
        }
      }
      i += len;
    } else {
      PixOrCopy v;
      if (use_color_cache && VP8LColorCacheContains(&hashers, argb[i])) {
        // push pixel as a color cache index
        const int idx = VP8LColorCacheGetIndex(&hashers, argb[i]);
        v = PixOrCopyCreateCacheIdx(idx);
      } else {
        if (use_color_cache) VP8LColorCacheInsert(&hashers, argb[i]);
        v = PixOrCopyCreateLiteral(argb[i]);
      }
      BackwardRefsCursorAdd(refs, v);
      if (i + 1 < pix_count) {
        HashChainInsert(hash_chain, &argb[i], i);
      }
      ++i;
    }
  }
  ok = !refs->error_;
 Error:
  if (cc_init) VP8LColorCacheClear(&hashers);
  return ok;
}

// Returns 1 on success.
static int BackwardReferencesTraceBackwards(int xsize, int ysize,
                                            const uint32_t* const argb,
                                            int quality, int cache_bits,
                                            VP8LHashChain* const hash_chain,
                                            VP8LBackwardRefs* const refs) {
  int ok = 0;
  const int dist_array_size = xsize * ysize;
  uint16_t* chosen_path = NULL;
  int chosen_path_size = 0;
  uint16_t* dist_array =
      (uint16_t*)WebPSafeMalloc(dist_array_size, sizeof(*dist_array));

  if (dist_array == NULL) goto Error;

  if (!BackwardReferencesHashChainDistanceOnly(
      xsize, ysize, argb, quality, cache_bits, hash_chain,
      refs, dist_array)) {
    goto Error;
  }
  TraceBackwards(dist_array, dist_array_size, &chosen_path, &chosen_path_size);
  if (!BackwardReferencesHashChainFollowChosenPath(
      xsize, ysize, argb, quality, cache_bits, chosen_path, chosen_path_size,
      hash_chain, refs)) {
    goto Error;
  }
  ok = 1;
 Error:
  WebPSafeFree(dist_array);
  return ok;
}

static void BackwardReferences2DLocality(int xsize,
                                         const VP8LBackwardRefs* const refs) {
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    if (PixOrCopyIsCopy(c.cur_pos)) {
      const int dist = c.cur_pos->argb_or_distance;
      const int transformed_dist = DistanceToPlaneCode(xsize, dist);
      c.cur_pos->argb_or_distance = transformed_dist;
    }
    VP8LRefsCursorNext(&c);
  }
}

// Returns entropy for the given cache bits.
static double ComputeCacheEntropy(const uint32_t* argb,
                                  const VP8LBackwardRefs* const refs,
                                  int cache_bits) {
  const int use_color_cache = (cache_bits > 0);
  int cc_init = 0;
  double entropy = MAX_ENTROPY;
  const double kSmallPenaltyForLargeCache = 4.0;
  VP8LColorCache hashers;
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  VP8LHistogram* histo = VP8LAllocateHistogram(cache_bits);
  if (histo == NULL) goto Error;

  if (use_color_cache) {
    cc_init = VP8LColorCacheInit(&hashers, cache_bits);
    if (!cc_init) goto Error;
  }
  if (!use_color_cache) {
    while (VP8LRefsCursorOk(&c)) {
      VP8LHistogramAddSinglePixOrCopy(histo, c.cur_pos);
      VP8LRefsCursorNext(&c);
    }
  } else {
    while (VP8LRefsCursorOk(&c)) {
      const PixOrCopy* const v = c.cur_pos;
      if (PixOrCopyIsLiteral(v)) {
        const uint32_t pix = *argb++;
        const uint32_t key = VP8LColorCacheGetIndex(&hashers, pix);
        if (VP8LColorCacheLookup(&hashers, key) == pix) {
          ++histo->literal_[NUM_LITERAL_CODES + NUM_LENGTH_CODES + key];
        } else {
          VP8LColorCacheSet(&hashers, key, pix);
          ++histo->blue_[pix & 0xff];
          ++histo->literal_[(pix >> 8) & 0xff];
          ++histo->red_[(pix >> 16) & 0xff];
          ++histo->alpha_[pix >> 24];
        }
      } else {
        int len = PixOrCopyLength(v);
        int code, extra_bits;
        VP8LPrefixEncodeBits(len, &code, &extra_bits);
        ++histo->literal_[NUM_LITERAL_CODES + code];
        VP8LPrefixEncodeBits(PixOrCopyDistance(v), &code, &extra_bits);
        ++histo->distance_[code];
        do {
          VP8LColorCacheInsert(&hashers, *argb++);
        } while(--len != 0);
      }
      VP8LRefsCursorNext(&c);
    }
  }
  entropy = VP8LHistogramEstimateBits(histo) +
      kSmallPenaltyForLargeCache * cache_bits;
 Error:
  if (cc_init) VP8LColorCacheClear(&hashers);
  VP8LFreeHistogram(histo);
  return entropy;
}

// Evaluate optimal cache bits for the local color cache.
// The input *best_cache_bits sets the maximum cache bits to use (passing 0
// implies disabling the local color cache). The local color cache is also
// disabled for the lower (<= 25) quality.
// Returns 0 in case of memory error.
static int CalculateBestCacheSize(const uint32_t* const argb,
                                  int xsize, int ysize, int quality,
                                  VP8LHashChain* const hash_chain,
                                  VP8LBackwardRefs* const refs,
                                  int* const lz77_computed,
                                  int* const best_cache_bits) {
  int eval_low = 1;
  int eval_high = 1;
  double entropy_low = MAX_ENTROPY;
  double entropy_high = MAX_ENTROPY;
  const double cost_mul = 5e-4;
  int cache_bits_low = 0;
  int cache_bits_high = (quality <= 25) ? 0 : *best_cache_bits;

  assert(cache_bits_high <= MAX_COLOR_CACHE_BITS);

  *lz77_computed = 0;
  if (cache_bits_high == 0) {
    *best_cache_bits = 0;
    // Local color cache is disabled.
    return 1;
  }
  if (!BackwardReferencesLz77(xsize, ysize, argb, cache_bits_low, quality, 0,
                              hash_chain, refs)) {
    return 0;
  }
  // Do a binary search to find the optimal entropy for cache_bits.
  while (eval_low || eval_high) {
    if (eval_low) {
      entropy_low = ComputeCacheEntropy(argb, refs, cache_bits_low);
      entropy_low += entropy_low * cache_bits_low * cost_mul;
      eval_low = 0;
    }
    if (eval_high) {
      entropy_high = ComputeCacheEntropy(argb, refs, cache_bits_high);
      entropy_high += entropy_high * cache_bits_high * cost_mul;
      eval_high = 0;
    }
    if (entropy_high < entropy_low) {
      const int prev_cache_bits_low = cache_bits_low;
      *best_cache_bits = cache_bits_high;
      cache_bits_low = (cache_bits_low + cache_bits_high) / 2;
      if (cache_bits_low != prev_cache_bits_low) eval_low = 1;
    } else {
      *best_cache_bits = cache_bits_low;
      cache_bits_high = (cache_bits_low + cache_bits_high) / 2;
      if (cache_bits_high != cache_bits_low) eval_high = 1;
    }
  }
  *lz77_computed = 1;
  return 1;
}

// Update (in-place) backward references for specified cache_bits.
static int BackwardRefsWithLocalCache(const uint32_t* const argb,
                                      int cache_bits,
                                      VP8LBackwardRefs* const refs) {
  int pixel_index = 0;
  VP8LColorCache hashers;
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  if (!VP8LColorCacheInit(&hashers, cache_bits)) return 0;

  while (VP8LRefsCursorOk(&c)) {
    PixOrCopy* const v = c.cur_pos;
    if (PixOrCopyIsLiteral(v)) {
      const uint32_t argb_literal = v->argb_or_distance;
      if (VP8LColorCacheContains(&hashers, argb_literal)) {
        const int ix = VP8LColorCacheGetIndex(&hashers, argb_literal);
        *v = PixOrCopyCreateCacheIdx(ix);
      } else {
        VP8LColorCacheInsert(&hashers, argb_literal);
      }
      ++pixel_index;
    } else {
      // refs was created without local cache, so it can not have cache indexes.
      int k;
      assert(PixOrCopyIsCopy(v));
      for (k = 0; k < v->len; ++k) {
        VP8LColorCacheInsert(&hashers, argb[pixel_index++]);
      }
    }
    VP8LRefsCursorNext(&c);
  }
  VP8LColorCacheClear(&hashers);
  return 1;
}

static VP8LBackwardRefs* GetBackwardReferencesLowEffort(
    int width, int height, const uint32_t* const argb, int quality,
    int* const cache_bits, VP8LHashChain* const hash_chain,
    VP8LBackwardRefs refs_array[2]) {
  VP8LBackwardRefs* refs_lz77 = &refs_array[0];
  *cache_bits = 0;
  if (!BackwardReferencesLz77(width, height, argb, 0, quality,
                              1 /* Low effort. */, hash_chain, refs_lz77)) {
    return NULL;
  }
  BackwardReferences2DLocality(width, refs_lz77);
  return refs_lz77;
}

static VP8LBackwardRefs* GetBackwardReferences(
    int width, int height, const uint32_t* const argb, int quality,
    int* const cache_bits, VP8LHashChain* const hash_chain,
    VP8LBackwardRefs refs_array[2]) {
  int lz77_is_useful;
  int lz77_computed;
  double bit_cost_lz77, bit_cost_rle;
  VP8LBackwardRefs* best = NULL;
  VP8LBackwardRefs* refs_lz77 = &refs_array[0];
  VP8LBackwardRefs* refs_rle = &refs_array[1];
  VP8LHistogram* histo = NULL;

  if (!CalculateBestCacheSize(argb, width, height, quality, hash_chain,
                              refs_lz77, &lz77_computed, cache_bits)) {
    goto Error;
  }

  if (lz77_computed) {
    // Transform refs_lz77 for the optimized cache_bits.
    if (*cache_bits > 0) {
      if (!BackwardRefsWithLocalCache(argb, *cache_bits, refs_lz77)) {
        goto Error;
      }
    }
  } else {
    if (!BackwardReferencesLz77(width, height, argb, *cache_bits, quality,
                                0 /* Low effort. */, hash_chain, refs_lz77)) {
      goto Error;
    }
  }

  if (!BackwardReferencesRle(width, height, argb, *cache_bits, refs_rle)) {
    goto Error;
  }

  histo = VP8LAllocateHistogram(*cache_bits);
  if (histo == NULL) goto Error;

  {
    // Evaluate LZ77 coding.
    VP8LHistogramCreate(histo, refs_lz77, *cache_bits);
    bit_cost_lz77 = VP8LHistogramEstimateBits(histo);
    // Evaluate RLE coding.
    VP8LHistogramCreate(histo, refs_rle, *cache_bits);
    bit_cost_rle = VP8LHistogramEstimateBits(histo);
    // Decide if LZ77 is useful.
    lz77_is_useful = (bit_cost_lz77 < bit_cost_rle);
  }

  // Choose appropriate backward reference.
  if (lz77_is_useful) {
    // TraceBackwards is costly. Don't execute it at lower quality.
    const int try_lz77_trace_backwards = (quality >= 25);
    best = refs_lz77;   // default guess: lz77 is better
    if (try_lz77_trace_backwards) {
      VP8LBackwardRefs* const refs_trace = refs_rle;
      if (!VP8LBackwardRefsCopy(refs_lz77, refs_trace)) {
        best = NULL;
        goto Error;
      }
      if (BackwardReferencesTraceBackwards(width, height, argb, quality,
                                           *cache_bits, hash_chain,
                                           refs_trace)) {
        double bit_cost_trace;
        // Evaluate LZ77 coding.
        VP8LHistogramCreate(histo, refs_trace, *cache_bits);
        bit_cost_trace = VP8LHistogramEstimateBits(histo);
        if (bit_cost_trace < bit_cost_lz77) {
          best = refs_trace;
        }
      }
    }
  } else {
    best = refs_rle;
  }

  BackwardReferences2DLocality(width, best);

 Error:
  VP8LFreeHistogram(histo);
  return best;
}

VP8LBackwardRefs* VP8LGetBackwardReferences(
    int width, int height, const uint32_t* const argb, int quality,
    int low_effort, int* const cache_bits, VP8LHashChain* const hash_chain,
    VP8LBackwardRefs refs_array[2]) {
  if (low_effort) {
    return GetBackwardReferencesLowEffort(width, height, argb, quality,
                                          cache_bits, hash_chain, refs_array);
  } else {
    return GetBackwardReferences(width, height, argb, quality, cache_bits,
                                 hash_chain, refs_array);
  }
}
