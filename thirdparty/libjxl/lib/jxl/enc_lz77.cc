// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lz77.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

namespace {

class SymbolCostEstimator {
 public:
  SymbolCostEstimator(size_t num_contexts, bool force_huffman,
                      const std::vector<std::vector<Token>>& tokens,
                      const LZ77Params& lz77) {
    std::vector<Histogram> builder(num_contexts);
    // Build histograms for estimating lz77 savings.
    HybridUintConfig uint_config;
    for (const auto& stream : tokens) {
      for (const auto& token : stream) {
        uint32_t tok, nbits, bits;
        (token.is_lz77_length ? lz77.length_uint_config : uint_config)
            .Encode(token.value, &tok, &nbits, &bits);
        tok += token.is_lz77_length ? lz77.min_symbol : 0;
        JXL_DASSERT(token.context < num_contexts);
        builder[token.context].Add(tok);
      }
    }
    max_alphabet_size_ = 0;
    for (size_t i = 0; i < num_contexts; i++) {
      max_alphabet_size_ =
          std::max(max_alphabet_size_, builder[i].counts.size());
    }
    bits_.resize(num_contexts * max_alphabet_size_);
    // TODO(veluca): SIMD?
    add_symbol_cost_.resize(num_contexts);
    for (size_t i = 0; i < num_contexts; i++) {
      float inv_total = 1.0f / (builder[i].total_count + 1e-8f);
      float total_cost = 0;
      for (size_t j = 0; j < builder[i].counts.size(); j++) {
        size_t cnt = builder[i].counts[j];
        float cost = 0;
        if (cnt != 0 && cnt != builder[i].total_count) {
          cost = -FastLog2f(cnt * inv_total);
          if (force_huffman) cost = std::ceil(cost);
        } else if (cnt == 0) {
          cost = ANS_LOG_TAB_SIZE;  // Highest possible cost.
        }
        bits_[i * max_alphabet_size_ + j] = cost;
        total_cost += cost * builder[i].counts[j];
      }
      // Penalty for adding a lz77 symbol to this contest (only used for static
      // cost model). Higher penalty for contexts that have a very low
      // per-symbol entropy.
      add_symbol_cost_[i] = std::max(0.0f, 6.0f - total_cost * inv_total);
    }
  }
  float Bits(size_t ctx, size_t sym) const {
    return bits_[ctx * max_alphabet_size_ + sym];
  }
  float LenCost(size_t ctx, size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    lz77.length_uint_config.Encode(len, &tok, &nbits, &bits);
    tok += lz77.min_symbol;
    return nbits + Bits(ctx, tok);
  }
  float DistCost(size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    HybridUintConfig().Encode(len, &tok, &nbits, &bits);
    return nbits + Bits(lz77.nonserialized_distance_context, tok);
  }
  float AddSymbolCost(size_t idx) const { return add_symbol_cost_[idx]; }

 private:
  size_t max_alphabet_size_;
  std::vector<float> bits_;
  std::vector<float> add_symbol_cost_;
};

std::vector<std::vector<Token>> ApplyLZ77_RLE(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  // TODO(veluca): tune heuristics here.
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77);
  float bit_decrease = 0;
  size_t total_symbols = 0;
  std::vector<float> sym_cost;
  HybridUintConfig uint_config;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    total_symbols += in.size();
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
      size_t num_to_copy = 0;
      size_t distance_symbol = 0;  // 1 for RLE.
      if (distance_multiplier != 0) {
        distance_symbol = 1;  // Special distance 1 if enabled.
        JXL_DASSERT(kSpecialDistances[1][0] == 1);
        JXL_DASSERT(kSpecialDistances[1][1] == 0);
      }
      if (i > 0) {
        for (; i + num_to_copy < in.size(); num_to_copy++) {
          if (in[i + num_to_copy].value != in[i - 1].value) {
            break;
          }
        }
      }
      if (num_to_copy == 0) {
        out.push_back(in[i]);
        continue;
      }
      float cost = sym_cost[i + num_to_copy] - sym_cost[i];
      // This subtraction might overflow, but that's OK.
      size_t lz77_len = num_to_copy - lz77.min_length;
      float lz77_cost = num_to_copy >= lz77.min_length
                            ? CeilLog2Nonzero(lz77_len + 1) + 1
                            : 0;
      if (num_to_copy < lz77.min_length || cost <= lz77_cost) {
        for (size_t j = 0; j < num_to_copy; j++) {
          out.push_back(in[i + j]);
        }
        i += num_to_copy - 1;
        continue;
      }
      // Output the LZ77 length
      out.emplace_back(in[i].context, static_cast<uint32_t>(lz77_len));
      out.back().is_lz77_length = true;
      i += num_to_copy - 1;
      bit_decrease += cost - lz77_cost;
      // Output the LZ77 copy distance.
      out.emplace_back(
          static_cast<uint32_t>(lz77.nonserialized_distance_context),
          static_cast<uint32_t>(distance_symbol));
    }
  }

  if (bit_decrease > total_symbols * 0.2 + 16) {
    return tokens_lz77;
  }
  return {};
}

// Hash chain for LZ77 matching
struct HashChain {
  size_t size_;
  std::vector<uint32_t> data_;

  unsigned hash_num_values_ = 32768;
  unsigned hash_mask_ = hash_num_values_ - 1;
  unsigned hash_shift_ = 5;

  std::vector<int> head;
  std::vector<uint32_t> chain;
  std::vector<int> val;

  // Speed up repetitions of zero
  std::vector<int> headz;
  std::vector<uint32_t> chainz;
  std::vector<uint32_t> zeros;
  uint32_t numzeros = 0;

  size_t window_size_;
  size_t window_mask_;
  size_t min_length_;
  size_t max_length_;

  // Map of special distance codes.
  std::unordered_map<int, int> special_dist_table_;
  size_t num_special_distances_ = 0;

  uint32_t maxchainlength = 256;  // window_size_ to allow all

  HashChain(const Token* data, size_t size, size_t window_size,
            size_t min_length, size_t max_length, size_t distance_multiplier)
      : size_(size),
        window_size_(window_size),
        window_mask_(window_size - 1),
        min_length_(min_length),
        max_length_(max_length) {
    data_.resize(size);
    for (size_t i = 0; i < size; i++) {
      data_[i] = data[i].value;
    }

    head.resize(hash_num_values_, -1);
    val.resize(window_size_, -1);
    chain.resize(window_size_);
    for (uint32_t i = 0; i < window_size_; ++i) {
      chain[i] = i;  // same value as index indicates uninitialized
    }

    zeros.resize(window_size_);
    headz.resize(window_size_ + 1, -1);
    chainz.resize(window_size_);
    for (uint32_t i = 0; i < window_size_; ++i) {
      chainz[i] = i;
    }
    // Translate distance to special distance code.
    if (distance_multiplier) {
      // Count down, so if due to small distance multiplier multiple distances
      // map to the same code, the smallest code will be used in the end.
      for (int i = kNumSpecialDistances - 1; i >= 0; --i) {
        special_dist_table_[SpecialDistance(i, distance_multiplier)] = i;
      }
      num_special_distances_ = kNumSpecialDistances;
    }
  }

  uint32_t GetHash(size_t pos) const {
    uint32_t result = 0;
    if (pos + 2 < size_) {
      // TODO(lode): take the MSB's of the uint32_t values into account as well,
      // given that the hash code itself is less than 32 bits.
      result ^= static_cast<uint32_t>(data_[pos + 0] << 0u);
      result ^= static_cast<uint32_t>(data_[pos + 1] << hash_shift_);
      result ^= static_cast<uint32_t>(data_[pos + 2] << (hash_shift_ * 2));
    } else {
      // No need to compute hash of last 2 bytes, the length 2 is too short.
      return 0;
    }
    return result & hash_mask_;
  }

  uint32_t CountZeros(size_t pos, uint32_t prevzeros) const {
    size_t end = pos + window_size_;
    if (end > size_) end = size_;
    if (prevzeros > 0) {
      if (prevzeros >= window_mask_ && data_[end - 1] == 0 &&
          end == pos + window_size_) {
        return prevzeros;
      } else {
        return prevzeros - 1;
      }
    }
    uint32_t num = 0;
    while (pos + num < end && data_[pos + num] == 0) num++;
    return num;
  }

  void Update(size_t pos) {
    uint32_t hashval = GetHash(pos);
    uint32_t wpos = pos & window_mask_;

    val[wpos] = static_cast<int>(hashval);
    if (head[hashval] != -1) chain[wpos] = head[hashval];
    head[hashval] = wpos;

    if (pos > 0 && data_[pos] != data_[pos - 1]) numzeros = 0;
    numzeros = CountZeros(pos, numzeros);

    zeros[wpos] = numzeros;
    if (headz[numzeros] != -1) chainz[wpos] = headz[numzeros];
    headz[numzeros] = wpos;
  }

  void Update(size_t pos, size_t len) {
    for (size_t i = 0; i < len; i++) {
      Update(pos + i);
    }
  }

  template <typename CB>
  void FindMatches(size_t pos, int max_dist, const CB& found_match) const {
    uint32_t wpos = pos & window_mask_;
    uint32_t hashval = GetHash(pos);
    uint32_t hashpos = chain[wpos];

    int prev_dist = 0;
    int end = std::min<int>(pos + max_length_, size_);
    uint32_t chainlength = 0;
    uint32_t best_len = 0;
    for (;;) {
      int dist = (hashpos <= wpos) ? (wpos - hashpos)
                                   : (wpos - hashpos + window_mask_ + 1);
      if (dist < prev_dist) break;
      prev_dist = dist;
      uint32_t len = 0;
      if (dist > 0) {
        int i = pos;
        int j = pos - dist;
        if (numzeros > 3) {
          int r = std::min<int>(numzeros - 1, zeros[hashpos]);
          if (i + r >= end) r = end - i - 1;
          i += r;
          j += r;
        }
        while (i < end && data_[i] == data_[j]) {
          i++;
          j++;
        }
        len = i - pos;
        // This can trigger even if the new length is slightly smaller than the
        // best length, because it is possible for a slightly cheaper distance
        // symbol to occur.
        if (len >= min_length_ && len + 2 >= best_len) {
          auto it = special_dist_table_.find(dist);
          int dist_symbol = (it == special_dist_table_.end())
                                ? (num_special_distances_ + dist - 1)
                                : it->second;
          found_match(len, dist_symbol);
          if (len > best_len) best_len = len;
        }
      }

      chainlength++;
      if (chainlength >= maxchainlength) break;

      if (numzeros >= 3 && len > numzeros) {
        if (hashpos == chainz[hashpos]) break;
        hashpos = chainz[hashpos];
        if (zeros[hashpos] != numzeros) break;
      } else {
        if (hashpos == chain[hashpos]) break;
        hashpos = chain[hashpos];
        if (val[hashpos] != static_cast<int>(hashval)) {
          // outdated hash value
          break;
        }
      }
    }
  }
  void FindMatch(size_t pos, int max_dist, size_t* result_dist_symbol,
                 size_t* result_len) const {
    *result_dist_symbol = 0;
    *result_len = 1;
    FindMatches(pos, max_dist, [&](size_t len, size_t dist_symbol) {
      if (len > *result_len ||
          (len == *result_len && *result_dist_symbol > dist_symbol)) {
        *result_len = len;
        *result_dist_symbol = dist_symbol;
      }
    });
  }
};

float LenCost(size_t len) {
  uint32_t nbits, bits, tok;
  HybridUintConfig(1, 0, 0).Encode(len, &tok, &nbits, &bits);
  constexpr float kCostTable[] = {
      2.797667318563126,  3.213177690381199,  2.5706009246743737,
      2.408392498667534,  2.829649191872326,  3.3923087753324577,
      4.029267451554331,  4.415576699706408,  4.509357574741465,
      9.21481543803004,   10.020590190114898, 11.858671627804766,
      12.45853300490526,  11.713105831990857, 12.561996324849314,
      13.775477692278367, 13.174027068768641,
  };
  size_t table_size = sizeof kCostTable / sizeof *kCostTable;
  if (tok >= table_size) tok = table_size - 1;
  return kCostTable[tok] + nbits;
}

// TODO(veluca): this does not take into account usage or non-usage of distance
// multipliers.
float DistCost(size_t dist) {
  uint32_t nbits, bits, tok;
  HybridUintConfig(7, 0, 0).Encode(dist, &tok, &nbits, &bits);
  constexpr float kCostTable[] = {
      6.368282626312716,  5.680793277090298,  8.347404197105247,
      7.641619201599141,  6.914328374119438,  7.959808291537444,
      8.70023120759855,   8.71378518934703,   9.379132523982769,
      9.110472749092708,  9.159029569270908,  9.430936766731973,
      7.278284055315169,  7.8278514904267755, 10.026641158289236,
      9.976049229827066,  9.64351607048908,   9.563403863480442,
      10.171474111762747, 10.45950155077234,  9.994813912104219,
      10.322524683741156, 8.465808729388186,  8.756254166066853,
      10.160930174662234, 10.247329273413435, 10.04090403724809,
      10.129398517544082, 9.342311691539546,  9.07608009102374,
      10.104799540677513, 10.378079384990906, 10.165828974075072,
      10.337595322341553, 7.940557464567944,  10.575665823319431,
      11.023344321751955, 10.736144698831827, 11.118277044595054,
      7.468468230648442,  10.738305230932939, 10.906980780216568,
      10.163468216353817, 10.17805759656433,  11.167283670483565,
      11.147050200274544, 10.517921919244333, 10.651764778156886,
      10.17074446448919,  11.217636876224745, 11.261630721139484,
      11.403140815247259, 10.892472096873417, 11.1859607804481,
      8.017346947551262,  7.895143720278828,  11.036577113822025,
      11.170562110315794, 10.326988722591086, 10.40872184751056,
      11.213498225466386, 11.30580635516863,  10.672272515665442,
      10.768069466228063, 11.145257364153565, 11.64668307145549,
      10.593156194627339, 11.207499484844943, 10.767517766396908,
      10.826629811407042, 10.737764794499988, 10.6200448518045,
      10.191315385198092, 8.468384171390085,  11.731295299170432,
      11.824619886654398, 10.41518844301179,  10.16310536548649,
      10.539423685097576, 10.495136599328031, 10.469112847728267,
      11.72057686174922,  10.910326337834674, 11.378921834673758,
      11.847759036098536, 11.92071647623854,  10.810628276345282,
      11.008601085273893, 11.910326337834674, 11.949212023423133,
      11.298614839104337, 11.611603659010392, 10.472930394619985,
      11.835564720850282, 11.523267392285337, 12.01055816679611,
      8.413029688994023,  11.895784139536406, 11.984679534970505,
      11.220654278717394, 11.716311684833672, 10.61036646226114,
      10.89849965960364,  10.203762898863669, 10.997560826267238,
      11.484217379438984, 11.792836176993665, 12.24310468755171,
      11.464858097919262, 12.212747017409377, 11.425595666074955,
      11.572048533398757, 12.742093965163013, 11.381874288645637,
      12.191870445817015, 11.683156920035426, 11.152442115262197,
      11.90303691580457,  11.653292787169159, 11.938615382266098,
      16.970641701570223, 16.853602280380002, 17.26240782594733,
      16.644655390108507, 17.14310889757499,  16.910935455445955,
      17.505678976959697, 17.213498225466388, 2.4162310293553024,
      3.494587244462329,  3.5258600986408344, 3.4959806589517095,
      3.098390886949687,  3.343454654302911,  3.588847442290287,
      4.14614790111827,   5.152948641990529,  7.433696808092598,
      9.716311684833672,
  };
  size_t table_size = sizeof kCostTable / sizeof *kCostTable;
  if (tok >= table_size) tok = table_size - 1;
  return kCostTable[tok] + nbits;
}

std::vector<std::vector<Token>> ApplyLZ77_LZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  // TODO(veluca): tune heuristics here.
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77);
  float bit_decrease = 0;
  size_t total_symbols = 0;
  HybridUintConfig uint_config;
  std::vector<float> sym_cost;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    total_symbols += in.size();
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }

    out.reserve(in.size());
    size_t max_distance = in.size();
    size_t min_length = lz77.min_length;
    JXL_DASSERT(min_length >= 3);
    size_t max_length = in.size();

    // Use next power of two as window size.
    size_t window_size = 1;
    while (window_size < max_distance && window_size < kWindowSize) {
      window_size <<= 1;
    }

    HashChain chain(in.data(), in.size(), window_size, min_length, max_length,
                    distance_multiplier);
    size_t len;
    size_t dist_symbol;

    const size_t max_lazy_match_len = 256;  // 0 to disable lazy matching

    // Whether the next symbol was already updated (to test lazy matching)
    bool already_updated = false;
    for (size_t i = 0; i < in.size(); i++) {
      out.push_back(in[i]);
      if (!already_updated) chain.Update(i);
      already_updated = false;
      chain.FindMatch(i, max_distance, &dist_symbol, &len);
      if (len >= min_length) {
        if (len < max_lazy_match_len && i + 1 < in.size()) {
          // Try length at next symbol lazy matching
          chain.Update(i + 1);
          already_updated = true;
          size_t len2, dist_symbol2;
          chain.FindMatch(i + 1, max_distance, &dist_symbol2, &len2);
          if (len2 > len) {
            // Use the lazy match. Add literal, and use the next length starting
            // from the next byte.
            ++i;
            already_updated = false;
            len = len2;
            dist_symbol = dist_symbol2;
            out.push_back(in[i]);
          }
        }

        float cost = sym_cost[i + len] - sym_cost[i];
        size_t lz77_len = len - lz77.min_length;
        float lz77_cost = LenCost(lz77_len) + DistCost(dist_symbol) +
                          sce.AddSymbolCost(out.back().context);

        if (lz77_cost <= cost) {
          out.back().value = len - min_length;
          out.back().is_lz77_length = true;
          out.emplace_back(
              static_cast<uint32_t>(lz77.nonserialized_distance_context),
              static_cast<uint32_t>(dist_symbol));
          bit_decrease += cost - lz77_cost;
        } else {
          // LZ77 match ignored, and symbol already pushed. Push all other
          // symbols and skip.
          for (size_t j = 1; j < len; j++) {
            out.push_back(in[i + j]);
          }
        }

        if (already_updated) {
          chain.Update(i + 2, len - 2);
          already_updated = false;
        } else {
          chain.Update(i + 1, len - 1);
        }
        i += len - 1;
      } else {
        // Literal, already pushed
      }
    }
  }

  if (bit_decrease > total_symbols * 0.2 + 16) {
    return tokens_lz77;
  }
  return {};
}

std::vector<std::vector<Token>> ApplyLZ77_Optimal(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  std::vector<std::vector<Token>> tokens_for_cost_estimate =
      ApplyLZ77_LZ77(params, num_contexts, tokens, lz77);
  // If greedy-LZ77 does not give better compression than no-lz77, no reason to
  // run the optimal matching.
  if (tokens_for_cost_estimate.empty()) return {};
  SymbolCostEstimator sce(num_contexts + 1, params.force_huffman,
                          tokens_for_cost_estimate, lz77);
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  HybridUintConfig uint_config;
  std::vector<float> sym_cost;
  std::vector<uint32_t> dist_symbols;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }

    out.reserve(in.size());
    size_t max_distance = in.size();
    size_t min_length = lz77.min_length;
    JXL_DASSERT(min_length >= 3);
    size_t max_length = in.size();

    // Use next power of two as window size.
    size_t window_size = 1;
    while (window_size < max_distance && window_size < kWindowSize) {
      window_size <<= 1;
    }

    HashChain chain(in.data(), in.size(), window_size, min_length, max_length,
                    distance_multiplier);

    struct MatchInfo {
      uint32_t len;
      uint32_t dist_symbol;
      uint32_t ctx;
      float total_cost = std::numeric_limits<float>::max();
    };
    // Total cost to encode the first N symbols.
    std::vector<MatchInfo> prefix_costs(in.size() + 1);
    prefix_costs[0].total_cost = 0;

    size_t rle_length = 0;
    size_t skip_lz77 = 0;
    for (size_t i = 0; i < in.size(); i++) {
      chain.Update(i);
      float lit_cost =
          prefix_costs[i].total_cost + sym_cost[i + 1] - sym_cost[i];
      if (prefix_costs[i + 1].total_cost > lit_cost) {
        prefix_costs[i + 1].dist_symbol = 0;
        prefix_costs[i + 1].len = 1;
        prefix_costs[i + 1].ctx = in[i].context;
        prefix_costs[i + 1].total_cost = lit_cost;
      }
      if (skip_lz77 > 0) {
        skip_lz77--;
        continue;
      }
      dist_symbols.clear();
      chain.FindMatches(i, max_distance,
                        [&dist_symbols](size_t len, size_t dist_symbol) {
                          if (dist_symbols.size() <= len) {
                            dist_symbols.resize(len + 1, dist_symbol);
                          }
                          if (dist_symbol < dist_symbols[len]) {
                            dist_symbols[len] = dist_symbol;
                          }
                        });
      if (dist_symbols.size() <= min_length) continue;
      {
        size_t best_cost = dist_symbols.back();
        for (size_t j = dist_symbols.size() - 1; j >= min_length; j--) {
          if (dist_symbols[j] < best_cost) {
            best_cost = dist_symbols[j];
          }
          dist_symbols[j] = best_cost;
        }
      }
      for (size_t j = min_length; j < dist_symbols.size(); j++) {
        // Cost model that uses results from lazy LZ77.
        float lz77_cost = sce.LenCost(in[i].context, j - min_length, lz77) +
                          sce.DistCost(dist_symbols[j], lz77);
        float cost = prefix_costs[i].total_cost + lz77_cost;
        if (prefix_costs[i + j].total_cost > cost) {
          prefix_costs[i + j].len = j;
          prefix_costs[i + j].dist_symbol = dist_symbols[j] + 1;
          prefix_costs[i + j].ctx = in[i].context;
          prefix_costs[i + j].total_cost = cost;
        }
      }
      // We are in a RLE sequence: skip all the symbols except the first 8 and
      // the last 8. This avoid quadratic costs for sequences with long runs of
      // the same symbol.
      if ((dist_symbols.back() == 0 && distance_multiplier == 0) ||
          (dist_symbols.back() == 1 && distance_multiplier != 0)) {
        rle_length++;
      } else {
        rle_length = 0;
      }
      if (rle_length >= 8 && dist_symbols.size() > 9) {
        skip_lz77 = dist_symbols.size() - 10;
        rle_length = 0;
      }
    }
    size_t pos = in.size();
    while (pos > 0) {
      bool is_lz77_length = prefix_costs[pos].dist_symbol != 0;
      if (is_lz77_length) {
        size_t dist_symbol = prefix_costs[pos].dist_symbol - 1;
        out.emplace_back(
            static_cast<uint32_t>(lz77.nonserialized_distance_context),
            static_cast<uint32_t>(dist_symbol));
      }
      uint32_t val =
          is_lz77_length
              ? (prefix_costs[pos].len - static_cast<uint32_t>(min_length))
              : in[pos - 1].value;
      out.emplace_back(prefix_costs[pos].ctx, val);
      out.back().is_lz77_length = is_lz77_length;
      pos -= prefix_costs[pos].len;
    }
    if (!out.empty()) std::reverse(out.begin(), out.end());
  }
  return tokens_lz77;
}

}  // namespace

std::vector<std::vector<Token>> ApplyLZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  switch (params.lz77_method) {
    case HistogramParams::LZ77Method::kRLE:
      return ApplyLZ77_RLE(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77:
      return ApplyLZ77_LZ77(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kOptimal:
      return ApplyLZ77_Optimal(params, num_contexts, tokens, lz77);
    default:
      return {};
  }
}

}  // namespace jxl
