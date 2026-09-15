// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_ans.h"

#include <jxl/memory_manager.h>

#include <cstdint>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_context_map.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {
namespace {

// Decodes a number in the range [0..255], by reading 1 - 11 bits.
inline int DecodeVarLenUint8(BitReader* input) {
  if (input->ReadFixedBits<1>()) {
    int nbits = static_cast<int>(input->ReadFixedBits<3>());
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

// Decodes a number in the range [0..65535], by reading 1 - 21 bits.
inline int DecodeVarLenUint16(BitReader* input) {
  if (input->ReadFixedBits<1>()) {
    int nbits = static_cast<int>(input->ReadFixedBits<4>());
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

Status ReadHistogram(int precision_bits, std::vector<int32_t>* counts,
                     BitReader* input) {
  int range = 1 << precision_bits;
  int simple_code = input->ReadBits(1);
  if (simple_code == 1) {
    int i;
    int symbols[2] = {0};
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(1) + 1;
    for (i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint8(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    counts->resize(max_symbol + 1);
    if (num_symbols == 1) {
      (*counts)[symbols[0]] = range;
    } else {
      if (symbols[0] == symbols[1]) {  // corrupt data
        return false;
      }
      (*counts)[symbols[0]] = input->ReadBits(precision_bits);
      (*counts)[symbols[1]] = range - (*counts)[symbols[0]];
    }
  } else {
    int is_flat = input->ReadBits(1);
    if (is_flat == 1) {
      int alphabet_size = DecodeVarLenUint8(input) + 1;
      JXL_ENSURE(alphabet_size <= range);
      *counts = CreateFlatHistogram(alphabet_size, range);
      return true;
    }

    uint32_t shift;
    {
      // TODO(veluca): speed up reading with table lookups.
      int upper_bound_log = FloorLog2Nonzero(ANS_LOG_TAB_SIZE + 1);
      int log = 0;
      for (; log < upper_bound_log; log++) {
        if (input->ReadFixedBits<1>() == 0) break;
      }
      shift = (input->ReadBits(log) | (1 << log)) - 1;
      if (shift > ANS_LOG_TAB_SIZE + 1) {
        return JXL_FAILURE("Invalid shift value");
      }
    }

    int length = DecodeVarLenUint8(input) + 3;
    counts->resize(length);
    int total_count = 0;

    static const uint8_t huff[128][2] = {
        {3, 10}, {7, 12}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {7, 13}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
        {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
        {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    };

    std::vector<int> logcounts(counts->size());
    int omit_log = -1;
    int omit_pos = -1;
    // This array remembers which symbols have an RLE length.
    std::vector<int> same(counts->size(), 0);
    for (size_t i = 0; i < logcounts.size(); ++i) {
      input->Refill();  // for PeekFixedBits + Advance
      int idx = input->PeekFixedBits<7>();
      input->Consume(huff[idx][0]);
      logcounts[i] = huff[idx][1];
      // The RLE symbol.
      if (logcounts[i] == ANS_LOG_TAB_SIZE + 1) {
        int rle_length = DecodeVarLenUint8(input);
        same[i] = rle_length + 5;
        i += rle_length + 3;
        continue;
      }
      if (logcounts[i] > omit_log) {
        omit_log = logcounts[i];
        omit_pos = i;
      }
    }
    // Invalid input, e.g. due to invalid usage of RLE.
    if (omit_pos < 0) return JXL_FAILURE("Invalid histogram.");
    if (static_cast<size_t>(omit_pos) + 1 < logcounts.size() &&
        logcounts[omit_pos + 1] == ANS_TAB_SIZE + 1) {
      return JXL_FAILURE("Invalid histogram.");
    }
    int prev = 0;
    int numsame = 0;
    for (size_t i = 0; i < logcounts.size(); ++i) {
      if (same[i]) {
        // RLE sequence, let this loop output the same count for the next
        // iterations.
        numsame = same[i] - 1;
        prev = i > 0 ? (*counts)[i - 1] : 0;
      }
      if (numsame > 0) {
        (*counts)[i] = prev;
        numsame--;
      } else {
        unsigned int code = logcounts[i];
        // omit_pos may not be negative at this point (checked before).
        if (i == static_cast<size_t>(omit_pos)) {
          continue;
        } else if (code == 0) {
          continue;
        } else if (code == 1) {
          (*counts)[i] = 1;
        } else {
          int bitcount = GetPopulationCountPrecision(code - 1, shift);
          (*counts)[i] = (1u << (code - 1)) +
                         (input->ReadBits(bitcount) << (code - 1 - bitcount));
        }
      }
      total_count += (*counts)[i];
    }
    (*counts)[omit_pos] = range - total_count;
    if ((*counts)[omit_pos] <= 0) {
      // The histogram we've read sums to more than total_count (including at
      // least 1 for the omitted value).
      return JXL_FAILURE("Invalid histogram count.");
    }
  }
  return true;
}

}  // namespace

Status DecodeANSCodes(JxlMemoryManager* memory_manager,
                      const size_t num_histograms,
                      const size_t max_alphabet_size, BitReader* in,
                      ANSCode* result) {
  result->memory_manager = memory_manager;
  result->degenerate_symbols.resize(num_histograms, -1);
  if (result->use_prefix_code) {
    JXL_ENSURE(max_alphabet_size <= 1 << PREFIX_MAX_BITS);
    result->huffman_data.resize(num_histograms);
    std::vector<uint16_t> alphabet_sizes(num_histograms);
    for (size_t c = 0; c < num_histograms; c++) {
      alphabet_sizes[c] = DecodeVarLenUint16(in) + 1;
      if (alphabet_sizes[c] > max_alphabet_size) {
        return JXL_FAILURE("Alphabet size is too long: %u", alphabet_sizes[c]);
      }
    }
    for (size_t c = 0; c < num_histograms; c++) {
      if (alphabet_sizes[c] > 1) {
        if (!result->huffman_data[c].ReadFromBitStream(alphabet_sizes[c], in)) {
          if (!in->AllReadsWithinBounds()) {
            return JXL_STATUS(StatusCode::kNotEnoughBytes,
                              "Not enough bytes for huffman code");
          }
          return JXL_FAILURE("Invalid huffman tree number %" PRIuS
                             ", alphabet size %u",
                             c, alphabet_sizes[c]);
        }
      } else {
        // 0-bit codes does not require extension tables.
        result->huffman_data[c].table_.clear();
        result->huffman_data[c].table_.resize(1u << kHuffmanTableBits);
      }
      for (const auto& h : result->huffman_data[c].table_) {
        if (h.bits <= kHuffmanTableBits) {
          result->UpdateMaxNumBits(c, h.value);
        }
      }
    }
  } else {
    JXL_ENSURE(max_alphabet_size <= ANS_MAX_ALPHABET_SIZE);
    size_t alloc_size = num_histograms * (1 << result->log_alpha_size) *
                        sizeof(AliasTable::Entry);
    JXL_ASSIGN_OR_RETURN(result->alias_tables,
                         AlignedMemory::Create(memory_manager, alloc_size));
    AliasTable::Entry* alias_tables =
        result->alias_tables.address<AliasTable::Entry>();
    for (size_t c = 0; c < num_histograms; ++c) {
      std::vector<int32_t> counts;
      if (!ReadHistogram(ANS_LOG_TAB_SIZE, &counts, in)) {
        return JXL_FAILURE("Invalid histogram bitstream.");
      }
      if (counts.size() > max_alphabet_size) {
        return JXL_FAILURE("Alphabet size is too long: %" PRIuS, counts.size());
      }
      while (!counts.empty() && counts.back() == 0) {
        counts.pop_back();
      }
      for (size_t s = 0; s < counts.size(); s++) {
        if (counts[s] != 0) {
          result->UpdateMaxNumBits(c, s);
        }
      }
      // InitAliasTable "fixes" empty counts to contain degenerate "0" symbol.
      int degenerate_symbol = counts.empty() ? 0 : (counts.size() - 1);
      for (int s = 0; s < degenerate_symbol; ++s) {
        if (counts[s] != 0) {
          degenerate_symbol = -1;
          break;
        }
      }
      result->degenerate_symbols[c] = degenerate_symbol;
      JXL_RETURN_IF_ERROR(
          InitAliasTable(counts, ANS_LOG_TAB_SIZE, result->log_alpha_size,
                         alias_tables + c * (1 << result->log_alpha_size)));
    }
  }
  return true;
}
Status DecodeUintConfig(size_t log_alpha_size, HybridUintConfig* uint_config,
                        BitReader* br) {
  br->Refill();
  size_t split_exponent = br->ReadBits(CeilLog2Nonzero(log_alpha_size + 1));
  size_t msb_in_token = 0;
  size_t lsb_in_token = 0;
  if (split_exponent != log_alpha_size) {
    // otherwise, msb/lsb don't matter.
    size_t nbits = CeilLog2Nonzero(split_exponent + 1);
    msb_in_token = br->ReadBits(nbits);
    if (msb_in_token > split_exponent) {
      // This could be invalid here already and we need to check this before
      // we use its value to read more bits.
      return JXL_FAILURE("Invalid HybridUintConfig");
    }
    nbits = CeilLog2Nonzero(split_exponent - msb_in_token + 1);
    lsb_in_token = br->ReadBits(nbits);
  }
  if (lsb_in_token + msb_in_token > split_exponent) {
    return JXL_FAILURE("Invalid HybridUintConfig");
  }
  *uint_config = HybridUintConfig(split_exponent, msb_in_token, lsb_in_token);
  return true;
}

Status DecodeUintConfigs(size_t log_alpha_size,
                         std::vector<HybridUintConfig>* uint_config,
                         BitReader* br) {
  // TODO(veluca): RLE?
  for (auto& cfg : *uint_config) {
    JXL_RETURN_IF_ERROR(DecodeUintConfig(log_alpha_size, &cfg, br));
  }
  return true;
}

LZ77Params::LZ77Params() { Bundle::Init(this); }
Status LZ77Params::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &enabled));
  if (!visitor->Conditional(enabled)) return true;
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(224), Val(512), Val(4096),
                                         BitsOffset(15, 8), 224, &min_symbol));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(3), Val(4), BitsOffset(2, 5),
                                         BitsOffset(8, 9), 3, &min_length));
  return true;
}

void ANSCode::UpdateMaxNumBits(size_t ctx, size_t symbol) {
  HybridUintConfig* cfg = &uint_config[ctx];
  // LZ77 symbols use a different uint config.
  if (lz77.enabled && lz77.nonserialized_distance_context != ctx &&
      symbol >= lz77.min_symbol) {
    symbol -= lz77.min_symbol;
    cfg = &lz77.length_uint_config;
  }
  size_t split_token = cfg->split_token;
  size_t msb_in_token = cfg->msb_in_token;
  size_t lsb_in_token = cfg->lsb_in_token;
  size_t split_exponent = cfg->split_exponent;
  if (symbol < split_token) {
    max_num_bits = std::max(max_num_bits, split_exponent);
    return;
  }
  uint32_t n_extra_bits =
      split_exponent - (msb_in_token + lsb_in_token) +
      ((symbol - split_token) >> (msb_in_token + lsb_in_token));
  size_t total_bits = msb_in_token + lsb_in_token + n_extra_bits + 1;
  max_num_bits = std::max(max_num_bits, total_bits);
}

Status DecodeHistograms(JxlMemoryManager* memory_manager, BitReader* br,
                        size_t num_contexts, ANSCode* code,
                        std::vector<uint8_t>* context_map, bool disallow_lz77) {
  JXL_RETURN_IF_ERROR(Bundle::Read(br, &code->lz77));
  if (code->lz77.enabled) {
    num_contexts++;
    JXL_RETURN_IF_ERROR(DecodeUintConfig(/*log_alpha_size=*/8,
                                         &code->lz77.length_uint_config, br));
  }
  if (code->lz77.enabled && disallow_lz77) {
    return JXL_FAILURE("Using LZ77 when explicitly disallowed");
  }
  size_t num_histograms = 1;
  context_map->resize(num_contexts);
  if (num_contexts > 1) {
    JXL_RETURN_IF_ERROR(
        DecodeContextMap(memory_manager, context_map, &num_histograms, br));
  }
  JXL_DEBUG_V(
      4, "Decoded context map of size %" PRIuS " and %" PRIuS " histograms",
      num_contexts, num_histograms);
  code->lz77.nonserialized_distance_context = context_map->back();
  code->use_prefix_code = static_cast<bool>(br->ReadFixedBits<1>());
  if (code->use_prefix_code) {
    code->log_alpha_size = PREFIX_MAX_BITS;
  } else {
    code->log_alpha_size = br->ReadFixedBits<2>() + 5;
  }
  code->uint_config.resize(num_histograms);
  JXL_RETURN_IF_ERROR(
      DecodeUintConfigs(code->log_alpha_size, &code->uint_config, br));
  const size_t max_alphabet_size = 1 << code->log_alpha_size;
  JXL_RETURN_IF_ERROR(DecodeANSCodes(memory_manager, num_histograms,
                                     max_alphabet_size, br, code));
  return true;
}

StatusOr<ANSSymbolReader> ANSSymbolReader::Create(const ANSCode* code,
                                                  BitReader* JXL_RESTRICT br,
                                                  size_t distance_multiplier) {
  AlignedMemory lz77_window_storage;
  if (code->lz77.enabled) {
    JxlMemoryManager* memory_manager = code->memory_manager;
    JXL_ASSIGN_OR_RETURN(
        lz77_window_storage,
        AlignedMemory::Create(memory_manager, kWindowSize * sizeof(uint32_t)));
  }
  return ANSSymbolReader(code, br, distance_multiplier,
                         std::move(lz77_window_storage));
}

ANSSymbolReader::ANSSymbolReader(const ANSCode* code,
                                 BitReader* JXL_RESTRICT br,
                                 size_t distance_multiplier,
                                 AlignedMemory&& lz77_window_storage)
    : alias_tables_(code->alias_tables.address<AliasTable::Entry>()),
      huffman_data_(code->huffman_data.data()),
      use_prefix_code_(code->use_prefix_code),
      configs(code->uint_config.data()),
      lz77_window_storage_(std::move(lz77_window_storage)) {
  if (!use_prefix_code_) {
    state_ = static_cast<uint32_t>(br->ReadFixedBits<32>());
    log_alpha_size_ = code->log_alpha_size;
    log_entry_size_ = ANS_LOG_TAB_SIZE - code->log_alpha_size;
    entry_size_minus_1_ = (1 << log_entry_size_) - 1;
  } else {
    state_ = (ANS_SIGNATURE << 16u);
  }
  if (!code->lz77.enabled) return;
  lz77_window_ = lz77_window_storage_.address<uint32_t>();
  lz77_ctx_ = code->lz77.nonserialized_distance_context;
  lz77_length_uint_ = code->lz77.length_uint_config;
  lz77_threshold_ = code->lz77.min_symbol;
  lz77_min_length_ = code->lz77.min_length;
  num_special_distances_ = distance_multiplier == 0 ? 0 : kNumSpecialDistances;
  for (size_t i = 0; i < num_special_distances_; i++) {
    special_distances_[i] = SpecialDistance(i, distance_multiplier);
  }
}

}  // namespace jxl
