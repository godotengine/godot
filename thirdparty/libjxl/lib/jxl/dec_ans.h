// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_ANS_H_
#define LIB_JXL_DEC_ANS_H_

// Library to decode the ANS population counts from the bit-stream and build a
// decoding table from them.

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_huffman.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

class ANSSymbolReader;

// Experiments show that best performance is typically achieved for a
// split-exponent of 3 or 4. Trend seems to be that '4' is better
// for large-ish pictures, and '3' better for rather small-ish pictures.
// This is plausible - the more special symbols we have, the better
// statistics we need to get a benefit out of them.

// Our hybrid-encoding scheme has dedicated tokens for the smallest
// (1 << split_exponents) numbers, and for the rest
// encodes (number of bits) + (msb_in_token sub-leading binary digits) +
// (lsb_in_token lowest binary digits) in the token, with the remaining bits
// then being encoded as data.
//
// Example with split_exponent = 4, msb_in_token = 2, lsb_in_token = 0.
//
// Numbers N in [0 .. 15]:
//   These get represented as (token=N, bits='').
// Numbers N >= 16:
//   If n is such that 2**n <= N < 2**(n+1),
//   and m = N - 2**n is the 'mantissa',
//   these get represented as:
// (token=split_token +
//        ((n - split_exponent) * 4) +
//        (m >> (n - msb_in_token)),
//  bits=m & (1 << (n - msb_in_token)) - 1)
// Specifically, we would get:
// N = 0 - 15:          (token=N, nbits=0, bits='')
// N = 16 (10000):      (token=16, nbits=2, bits='00')
// N = 17 (10001):      (token=16, nbits=2, bits='01')
// N = 20 (10100):      (token=17, nbits=2, bits='00')
// N = 24 (11000):      (token=18, nbits=2, bits='00')
// N = 28 (11100):      (token=19, nbits=2, bits='00')
// N = 32 (100000):     (token=20, nbits=3, bits='000')
// N = 65535:           (token=63, nbits=13, bits='1111111111111')
struct HybridUintConfig {
  uint32_t split_exponent;
  uint32_t split_token;
  uint32_t msb_in_token;
  uint32_t lsb_in_token;
  JXL_INLINE void Encode(uint32_t value, uint32_t* JXL_RESTRICT token,
                         uint32_t* JXL_RESTRICT nbits,
                         uint32_t* JXL_RESTRICT bits) const {
    if (value < split_token) {
      *token = value;
      *nbits = 0;
      *bits = 0;
    } else {
      uint32_t n = FloorLog2Nonzero(value);
      uint32_t m = value - (1 << n);
      *token = split_token +
               ((n - split_exponent) << (msb_in_token + lsb_in_token)) +
               ((m >> (n - msb_in_token)) << lsb_in_token) +
               (m & ((1 << lsb_in_token) - 1));
      *nbits = n - msb_in_token - lsb_in_token;
      *bits = (value >> lsb_in_token) & ((1UL << *nbits) - 1);
    }
  }

  explicit HybridUintConfig(uint32_t split_exponent = 4,
                            uint32_t msb_in_token = 2,
                            uint32_t lsb_in_token = 0)
      : split_exponent(split_exponent),
        split_token(1 << split_exponent),
        msb_in_token(msb_in_token),
        lsb_in_token(lsb_in_token) {
    JXL_DASSERT(split_exponent >= msb_in_token + lsb_in_token);
  }
};

struct LZ77Params : public Fields {
  LZ77Params();
  JXL_FIELDS_NAME(LZ77Params)
  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;
  bool enabled;

  // Symbols above min_symbol use a special hybrid uint encoding and
  // represent a length, to be added to min_length.
  uint32_t min_symbol;
  uint32_t min_length;

  // Not serialized by VisitFields.
  HybridUintConfig length_uint_config{0, 0, 0};

  size_t nonserialized_distance_context;
};

static constexpr size_t kWindowSize = 1 << 20;
static constexpr size_t kNumSpecialDistances = 120;
// Table of special distance codes from WebP lossless.
static constexpr int8_t kSpecialDistances[kNumSpecialDistances][2] = {
    {0, 1},  {1, 0},  {1, 1},  {-1, 1}, {0, 2},  {2, 0},  {1, 2},  {-1, 2},
    {2, 1},  {-2, 1}, {2, 2},  {-2, 2}, {0, 3},  {3, 0},  {1, 3},  {-1, 3},
    {3, 1},  {-3, 1}, {2, 3},  {-2, 3}, {3, 2},  {-3, 2}, {0, 4},  {4, 0},
    {1, 4},  {-1, 4}, {4, 1},  {-4, 1}, {3, 3},  {-3, 3}, {2, 4},  {-2, 4},
    {4, 2},  {-4, 2}, {0, 5},  {3, 4},  {-3, 4}, {4, 3},  {-4, 3}, {5, 0},
    {1, 5},  {-1, 5}, {5, 1},  {-5, 1}, {2, 5},  {-2, 5}, {5, 2},  {-5, 2},
    {4, 4},  {-4, 4}, {3, 5},  {-3, 5}, {5, 3},  {-5, 3}, {0, 6},  {6, 0},
    {1, 6},  {-1, 6}, {6, 1},  {-6, 1}, {2, 6},  {-2, 6}, {6, 2},  {-6, 2},
    {4, 5},  {-4, 5}, {5, 4},  {-5, 4}, {3, 6},  {-3, 6}, {6, 3},  {-6, 3},
    {0, 7},  {7, 0},  {1, 7},  {-1, 7}, {5, 5},  {-5, 5}, {7, 1},  {-7, 1},
    {4, 6},  {-4, 6}, {6, 4},  {-6, 4}, {2, 7},  {-2, 7}, {7, 2},  {-7, 2},
    {3, 7},  {-3, 7}, {7, 3},  {-7, 3}, {5, 6},  {-5, 6}, {6, 5},  {-6, 5},
    {8, 0},  {4, 7},  {-4, 7}, {7, 4},  {-7, 4}, {8, 1},  {8, 2},  {6, 6},
    {-6, 6}, {8, 3},  {5, 7},  {-5, 7}, {7, 5},  {-7, 5}, {8, 4},  {6, 7},
    {-6, 7}, {7, 6},  {-7, 6}, {8, 5},  {7, 7},  {-7, 7}, {8, 6},  {8, 7}};
static JXL_INLINE int SpecialDistance(size_t index, int multiplier) {
  int dist = kSpecialDistances[index][0] +
             static_cast<int>(multiplier) * kSpecialDistances[index][1];
  return (dist > 1) ? dist : 1;
}

struct ANSCode {
  AlignedMemory alias_tables;
  std::vector<HuffmanDecodingData> huffman_data;
  std::vector<HybridUintConfig> uint_config;
  std::vector<int> degenerate_symbols;
  bool use_prefix_code;
  uint8_t log_alpha_size;  // for ANS.
  LZ77Params lz77;
  // Maximum number of bits necessary to represent the result of a
  // ReadHybridUint call done with this ANSCode.
  size_t max_num_bits = 0;
  JxlMemoryManager* memory_manager;
  void UpdateMaxNumBits(size_t ctx, size_t symbol);
};

class ANSSymbolReader {
 public:
  // Invalid symbol reader, to be overwritten.
  ANSSymbolReader() = default;
  static StatusOr<ANSSymbolReader> Create(const ANSCode* code,
                                          BitReader* JXL_RESTRICT br,
                                          size_t distance_multiplier = 0);

  JXL_INLINE size_t ReadSymbolANSWithoutRefill(const size_t histo_idx,
                                               BitReader* JXL_RESTRICT br) {
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1u);

    const AliasTable::Entry* table =
        &alias_tables_[histo_idx << log_alpha_size_];
    const AliasTable::Symbol symbol =
        AliasTable::Lookup(table, res, log_entry_size_, entry_size_minus_1_);
    state_ = symbol.freq * (state_ >> ANS_LOG_TAB_SIZE) + symbol.offset;

#if JXL_TRUE
    // Branchless version is about equally fast on SKX.
    const uint32_t new_state =
        (state_ << 16u) | static_cast<uint32_t>(br->PeekFixedBits<16>());
    const bool normalize = state_ < (1u << 16u);
    state_ = normalize ? new_state : state_;
    br->Consume(normalize ? 16 : 0);
#else
    if (JXL_UNLIKELY(state_ < (1u << 16u))) {
      state_ = (state_ << 16u) | br->PeekFixedBits<16>();
      br->Consume(16);
    }
#endif
    const uint32_t next_res = state_ & (ANS_TAB_SIZE - 1u);
    AliasTable::Prefetch(table, next_res, log_entry_size_);

    return symbol.value;
  }

  JXL_INLINE size_t ReadSymbolHuffWithoutRefill(const size_t histo_idx,
                                                BitReader* JXL_RESTRICT br) {
    return huffman_data_[histo_idx].ReadSymbol(br);
  }

  JXL_INLINE size_t ReadSymbolWithoutRefill(const size_t histo_idx,
                                            BitReader* JXL_RESTRICT br) {
    // TODO(veluca): hoist if in hotter loops.
    if (JXL_UNLIKELY(use_prefix_code_)) {
      return ReadSymbolHuffWithoutRefill(histo_idx, br);
    }
    return ReadSymbolANSWithoutRefill(histo_idx, br);
  }

  JXL_INLINE size_t ReadSymbol(const size_t histo_idx,
                               BitReader* JXL_RESTRICT br) {
    br->Refill();
    return ReadSymbolWithoutRefill(histo_idx, br);
  }

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  bool CheckANSFinalState() const { return true; }
#else
  bool CheckANSFinalState() const { return state_ == (ANS_SIGNATURE << 16u); }
#endif

  template <typename BitReader>
  static JXL_INLINE uint32_t ReadHybridUintConfig(
      const HybridUintConfig& config, size_t token, BitReader* br) {
    size_t split_token = config.split_token;
    size_t msb_in_token = config.msb_in_token;
    size_t lsb_in_token = config.lsb_in_token;
    size_t split_exponent = config.split_exponent;
    // Fast-track version of hybrid integer decoding.
    if (token < split_token) return token;
    uint32_t nbits = split_exponent - (msb_in_token + lsb_in_token) +
                     ((token - split_token) >> (msb_in_token + lsb_in_token));
    // Max amount of bits for ReadBits is 32 and max valid left shift is 29
    // bits. However, for speed no error is propagated here, instead limit the
    // nbits size. If nbits > 29, the code stream is invalid, but no error is
    // returned.
    // Note that in most cases we will emit an error if the histogram allows
    // representing numbers that would cause invalid shifts, but we need to
    // keep this check as when LZ77 is enabled it might make sense to have an
    // histogram that could in principle cause invalid shifts.
    nbits &= 31u;
    uint32_t low = token & ((1 << lsb_in_token) - 1);
    token >>= lsb_in_token;
    const size_t bits = br->PeekBits(nbits);
    br->Consume(nbits);
    size_t ret = (((((1 << msb_in_token) | (token & ((1 << msb_in_token) - 1)))
                    << nbits) |
                   bits)
                  << lsb_in_token) |
                 low;
    // TODO(eustas): mark BitReader as unhealthy if nbits > 29 or ret does not
    //               fit uint32_t
    return static_cast<uint32_t>(ret);
  }

  // Takes a *clustered* idx. Can only use if HuffRleOnly() is true.
  JXL_INLINE void ReadHybridUintClusteredHuffRleOnly(size_t ctx,
                                                     BitReader* JXL_RESTRICT br,
                                                     uint32_t* value,
                                                     uint32_t* run) {
    JXL_DASSERT(IsHuffRleOnly());
    br->Refill();  // covers ReadSymbolWithoutRefill + PeekBits
    size_t token = ReadSymbolHuffWithoutRefill(ctx, br);
    if (JXL_UNLIKELY(token >= lz77_threshold_)) {
      *run =
          ReadHybridUintConfig(lz77_length_uint_, token - lz77_threshold_, br) +
          lz77_min_length_ - 1;
      return;
    }
    *value = ReadHybridUintConfig(configs[ctx], token, br);
  }
  bool IsHuffRleOnly() const {
    if (lz77_window_ == nullptr) return false;
    if (!use_prefix_code_) return false;
    for (size_t i = 0; i < kHuffmanTableBits; i++) {
      if (huffman_data_[lz77_ctx_].table_[i].bits) return false;
      if (huffman_data_[lz77_ctx_].table_[i].value != 1) return false;
    }
    if (configs[lz77_ctx_].split_token > 1) return false;
    return true;
  }
  bool UsesLZ77() { return lz77_window_ != nullptr; }

  // Takes a *clustered* idx. Inlined, for use in hot paths.
  template <bool uses_lz77>
  JXL_INLINE size_t ReadHybridUintClusteredInlined(size_t ctx,
                                                   BitReader* JXL_RESTRICT br) {
    if (uses_lz77) {
      if (JXL_UNLIKELY(num_to_copy_ > 0)) {
        size_t ret = lz77_window_[(copy_pos_++) & kWindowMask];
        num_to_copy_--;
        lz77_window_[(num_decoded_++) & kWindowMask] = ret;
        return ret;
      }
    }

    br->Refill();  // covers ReadSymbolWithoutRefill + PeekBits
    size_t token = ReadSymbolWithoutRefill(ctx, br);
    if (uses_lz77) {
      if (JXL_UNLIKELY(token >= lz77_threshold_)) {
        num_to_copy_ = ReadHybridUintConfig(lz77_length_uint_,
                                            token - lz77_threshold_, br) +
                       lz77_min_length_;
        br->Refill();  // covers ReadSymbolWithoutRefill + PeekBits
        // Distance code.
        size_t token = ReadSymbolWithoutRefill(lz77_ctx_, br);
        size_t distance = ReadHybridUintConfig(configs[lz77_ctx_], token, br);
        if (JXL_LIKELY(distance < num_special_distances_)) {
          distance = special_distances_[distance];
        } else {
          distance = distance + 1 - num_special_distances_;
        }
        if (JXL_UNLIKELY(distance > num_decoded_)) {
          distance = num_decoded_;
        }
        if (JXL_UNLIKELY(distance > kWindowSize)) {
          distance = kWindowSize;
        }
        copy_pos_ = num_decoded_ - distance;
        if (JXL_UNLIKELY(distance == 0)) {
          JXL_DASSERT(lz77_window_ != nullptr);
          // distance 0 -> num_decoded_ == copy_pos_ == 0
          size_t to_fill = std::min<size_t>(num_to_copy_, kWindowSize);
          memset(lz77_window_, 0, to_fill * sizeof(lz77_window_[0]));
        }
        // TODO(eustas): overflow; mark BitReader as unhealthy
        if (num_to_copy_ < lz77_min_length_) return 0;
        // the code below is the same as doing this:
        //        return ReadHybridUintClustered<uses_lz77>(ctx, br);
        // but gcc doesn't like recursive inlining

        size_t ret = lz77_window_[(copy_pos_++) & kWindowMask];
        num_to_copy_--;
        lz77_window_[(num_decoded_++) & kWindowMask] = ret;
        return ret;
      }
    }
    size_t ret = ReadHybridUintConfig(configs[ctx], token, br);
    if (uses_lz77 && lz77_window_)
      lz77_window_[(num_decoded_++) & kWindowMask] = ret;
    return ret;
  }

  // same but not inlined
  template <bool uses_lz77>
  size_t ReadHybridUintClustered(size_t ctx, BitReader* JXL_RESTRICT br) {
    return ReadHybridUintClusteredInlined<uses_lz77>(ctx, br);
  }

  // inlined only in the no-lz77 case
  template <bool uses_lz77>
  JXL_INLINE size_t
  ReadHybridUintClusteredMaybeInlined(size_t ctx, BitReader* JXL_RESTRICT br) {
    if (uses_lz77) {
      return ReadHybridUintClustered<uses_lz77>(ctx, br);
    } else {
      return ReadHybridUintClusteredInlined<uses_lz77>(ctx, br);
    }
  }

  // inlined, for use in hot paths
  template <bool uses_lz77>
  JXL_INLINE size_t
  ReadHybridUintInlined(size_t ctx, BitReader* JXL_RESTRICT br,
                        const std::vector<uint8_t>& context_map) {
    return ReadHybridUintClustered<uses_lz77>(context_map[ctx], br);
  }

  // not inlined, for use in non-hot paths
  size_t ReadHybridUint(size_t ctx, BitReader* JXL_RESTRICT br,
                        const std::vector<uint8_t>& context_map) {
    return ReadHybridUintClustered</*uses_lz77=*/true>(context_map[ctx], br);
  }

  // ctx is a *clustered* context!
  // This function will modify the ANS state as if `count` symbols have been
  // decoded.
  bool IsSingleValueAndAdvance(size_t ctx, uint32_t* value, size_t count) {
    // TODO(veluca): No optimization for Huffman mode yet.
    if (use_prefix_code_) return false;
    // TODO(eustas): propagate "degenerate_symbol" to simplify this method.
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1u);
    const AliasTable::Entry* table = &alias_tables_[ctx << log_alpha_size_];
    AliasTable::Symbol symbol =
        AliasTable::Lookup(table, res, log_entry_size_, entry_size_minus_1_);
    if (symbol.freq != ANS_TAB_SIZE) return false;
    if (configs[ctx].split_token <= symbol.value) return false;
    if (symbol.value >= lz77_threshold_) return false;
    *value = symbol.value;
    if (lz77_window_) {
      for (size_t i = 0; i < count; i++) {
        lz77_window_[(num_decoded_++) & kWindowMask] = symbol.value;
      }
    }
    return true;
  }

  static constexpr size_t kMaxCheckpointInterval = 512;
  struct Checkpoint {
    uint32_t state;
    uint32_t num_to_copy;
    uint32_t copy_pos;
    uint32_t num_decoded;
    uint32_t lz77_window[kMaxCheckpointInterval];
  };
  void Save(Checkpoint* checkpoint) {
    checkpoint->state = state_;
    checkpoint->num_decoded = num_decoded_;
    checkpoint->num_to_copy = num_to_copy_;
    checkpoint->copy_pos = copy_pos_;
    if (lz77_window_) {
      size_t win_start = num_decoded_ & kWindowMask;
      size_t win_end = (num_decoded_ + kMaxCheckpointInterval) & kWindowMask;
      if (win_end > win_start) {
        memcpy(checkpoint->lz77_window, lz77_window_ + win_start,
               (win_end - win_start) * sizeof(*lz77_window_));
      } else {
        memcpy(checkpoint->lz77_window, lz77_window_ + win_start,
               (kWindowSize - win_start) * sizeof(*lz77_window_));
        memcpy(checkpoint->lz77_window + (kWindowSize - win_start),
               lz77_window_, win_end * sizeof(*lz77_window_));
      }
    }
  }
  void Restore(const Checkpoint& checkpoint) {
    state_ = checkpoint.state;
    JXL_DASSERT(num_decoded_ <=
                checkpoint.num_decoded + kMaxCheckpointInterval);
    num_decoded_ = checkpoint.num_decoded;
    num_to_copy_ = checkpoint.num_to_copy;
    copy_pos_ = checkpoint.copy_pos;
    if (lz77_window_) {
      size_t win_start = num_decoded_ & kWindowMask;
      size_t win_end = (num_decoded_ + kMaxCheckpointInterval) & kWindowMask;
      if (win_end > win_start) {
        memcpy(lz77_window_ + win_start, checkpoint.lz77_window,
               (win_end - win_start) * sizeof(*lz77_window_));
      } else {
        memcpy(lz77_window_ + win_start, checkpoint.lz77_window,
               (kWindowSize - win_start) * sizeof(*lz77_window_));
        memcpy(lz77_window_, checkpoint.lz77_window + (kWindowSize - win_start),
               win_end * sizeof(*lz77_window_));
      }
    }
  }

 private:
  ANSSymbolReader(const ANSCode* code, BitReader* JXL_RESTRICT br,
                  size_t distance_multiplier,
                  AlignedMemory&& lz77_window_storage);

  const AliasTable::Entry* JXL_RESTRICT alias_tables_;  // not owned
  const HuffmanDecodingData* huffman_data_;
  bool use_prefix_code_;
  uint32_t state_ = ANS_SIGNATURE << 16u;
  const HybridUintConfig* JXL_RESTRICT configs;
  uint32_t log_alpha_size_{};
  uint32_t log_entry_size_{};
  uint32_t entry_size_minus_1_{};

  // LZ77 structures and constants.
  static constexpr size_t kWindowMask = kWindowSize - 1;
  // a std::vector incurs unacceptable decoding speed loss because of
  // initialization.
  AlignedMemory lz77_window_storage_;
  uint32_t* lz77_window_ = nullptr;
  uint32_t num_decoded_ = 0;
  uint32_t num_to_copy_ = 0;
  uint32_t copy_pos_ = 0;
  uint32_t lz77_ctx_ = 0;
  uint32_t lz77_min_length_ = 0;
  uint32_t lz77_threshold_ = 1 << 20;  // bigger than any symbol.
  HybridUintConfig lz77_length_uint_;
  uint32_t special_distances_[kNumSpecialDistances]{};
  uint32_t num_special_distances_{};
};

Status DecodeHistograms(JxlMemoryManager* memory_manager, BitReader* br,
                        size_t num_contexts, ANSCode* code,
                        std::vector<uint8_t>* context_map,
                        bool disallow_lz77 = false);

// Exposed for tests.
Status DecodeUintConfigs(size_t log_alpha_size,
                         std::vector<HybridUintConfig>* uint_config,
                         BitReader* br);

}  // namespace jxl

#endif  // LIB_JXL_DEC_ANS_H_
