// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ans.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_ans_simd.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_huffman.h"
#include "lib/jxl/enc_lz77.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/simd_util.h"

namespace jxl {

namespace {

#if (!JXL_IS_DEBUG_BUILD)
constexpr
#endif
    bool ans_fuzzer_friendly_ = false;

const int kMaxNumSymbolsForSmallCode = 2;

template <typename Writer>
void StoreVarLenUint8(size_t n, Writer* writer) {
  JXL_DASSERT(n <= 255);
  if (n == 0) {
    writer->Write(1, 0);
  } else {
    writer->Write(1, 1);
    size_t nbits = FloorLog2Nonzero(n);
    writer->Write(3, nbits);
    writer->Write(nbits, n - (1ULL << nbits));
  }
}

template <typename Writer>
void StoreVarLenUint16(size_t n, Writer* writer) {
  JXL_DASSERT(n <= 65535);
  if (n == 0) {
    writer->Write(1, 0);
  } else {
    writer->Write(1, 1);
    size_t nbits = FloorLog2Nonzero(n);
    writer->Write(4, nbits);
    writer->Write(nbits, n - (1ULL << nbits));
  }
}

class ANSEncodingHistogram {
 public:
  const std::vector<ANSHistBin>& Counts() const { return counts_; }
  float Cost() const { return cost_; }
  // The only way to construct valid histogram for ANS encoding
  static StatusOr<ANSEncodingHistogram> ComputeBest(
      const Histogram& histo,
      HistogramParams::ANSHistogramStrategy ans_histogram_strategy) {
    ANSEncodingHistogram result;

    result.alphabet_size_ = histo.alphabet_size();
    if (result.alphabet_size_ > ANS_MAX_ALPHABET_SIZE)
      return JXL_FAILURE("Too many entries in an ANS histogram");

    if (result.alphabet_size_ > 0) {
      // Flat code
      result.method_ = 0;
      result.num_symbols_ = result.alphabet_size_;
      result.counts_ = CreateFlatHistogram(result.alphabet_size_, ANS_TAB_SIZE);
      // in this case length can be non-suitable for SIMD - fix it
      result.counts_.resize(histo.counts.size());
      SizeWriter writer;
      JXL_RETURN_IF_ERROR(result.Encode(&writer));
      result.cost_ = writer.size + EstimateDataBitsFlat(histo);
    } else {
      // Empty histogram
      result.method_ = 1;
      result.num_symbols_ = 0;
      result.cost_ = 3;
      return result;
    }

    size_t symbol_count = 0;
    for (size_t n = 0; n < result.alphabet_size_; ++n) {
      if (histo.counts[n] > 0) {
        if (symbol_count < kMaxNumSymbolsForSmallCode) {
          result.symbols_[symbol_count] = n;
        }
        ++symbol_count;
      }
    }
    result.num_symbols_ = symbol_count;
    if (symbol_count == 1) {
      // Single-bin histogram
      result.method_ = 1;
      result.counts_ = histo.counts;
      result.counts_[result.symbols_[0]] = ANS_TAB_SIZE;
      SizeWriter writer;
      JXL_RETURN_IF_ERROR(result.Encode(&writer));
      result.cost_ = writer.size;
      return result;
    }

    // Here min 2 symbols
    ANSEncodingHistogram normalized = result;
    auto try_shift = [&](uint32_t shift) -> Status {
      // `shift = 12` and `shift = 11` are the same
      normalized.method_ = std::min(shift, ANS_LOG_TAB_SIZE - 1) + 1;

      if (!normalized.RebalanceHistogram(histo)) {
        return JXL_FAILURE("Logic error: couldn't rebalance a histogram");
      }
      SizeWriter writer;
      JXL_RETURN_IF_ERROR(normalized.Encode(&writer));
      normalized.cost_ = writer.size + normalized.EstimateDataBits(histo);
      if (normalized.cost_ < result.cost_) {
        result = normalized;
      }
      return true;
    };

    switch (ans_histogram_strategy) {
      case HistogramParams::ANSHistogramStrategy::kPrecise:
        for (uint32_t shift = 0; shift < ANS_LOG_TAB_SIZE; shift++) {
          JXL_RETURN_IF_ERROR(try_shift(shift));
        }
        break;
      case HistogramParams::ANSHistogramStrategy::kApproximate:
        for (uint32_t shift = 0; shift <= ANS_LOG_TAB_SIZE; shift += 2) {
          JXL_RETURN_IF_ERROR(try_shift(shift));
        }
        break;
      case HistogramParams::ANSHistogramStrategy::kFast:
        JXL_RETURN_IF_ERROR(try_shift(0));
        JXL_RETURN_IF_ERROR(try_shift(ANS_LOG_TAB_SIZE / 2));
        JXL_RETURN_IF_ERROR(try_shift(ANS_LOG_TAB_SIZE));
        break;
    }

      // Sanity check
#if JXL_IS_DEBUG_BUILD
    JXL_ENSURE(histo.counts.size() == result.counts_.size());
    ANSHistBin total = 0;  // Used only in assert.
    for (size_t i = 0; i < result.alphabet_size_; ++i) {
      JXL_ENSURE(result.counts_[i] >= 0);
      // For non-flat histogram values should be zero or non-zero simultaneously
      // for the same symbol in both initial and normalized histograms.
      JXL_ENSURE(result.method_ == 0 ||
                 (histo.counts[i] > 0) == (result.counts_[i] > 0));
      // Check accuracy of the histogram values
      if (result.method_ > 0 && result.counts_[i] > 0 &&
          i != result.omit_pos_) {
        int logcounts = FloorLog2Nonzero<uint32_t>(result.counts_[i]);
        int bitcount =
            GetPopulationCountPrecision(logcounts, result.method_ - 1);
        int drop_bits = logcounts - bitcount;
        // Check that the value is divisible by 2^drop_bits
        JXL_ENSURE((result.counts_[i] & ((1 << drop_bits) - 1)) == 0);
      }
      total += result.counts_[i];
    }
    for (size_t i = result.alphabet_size_; i < result.counts_.size(); ++i) {
      JXL_ENSURE(histo.counts[i] == 0);
      JXL_ENSURE(result.counts_[i] == 0);
    }
    JXL_ENSURE((histo.total_count == 0) || (total == ANS_TAB_SIZE));
#endif
    return result;
  }

  template <typename Writer>
  Status Encode(Writer* writer) {
    // The check ensures also that all RLE sequences can be
    // encoded by `StoreVarLenUint8`
    JXL_ENSURE(alphabet_size_ <= ANS_MAX_ALPHABET_SIZE);

    /// Flat histogram.
    if (method_ == 0) {
      // Mark non-small tree.
      writer->Write(1, 0);
      // Mark uniform histogram.
      writer->Write(1, 1);
      JXL_ENSURE(alphabet_size_ > 0);
      // Encode alphabet size.
      StoreVarLenUint8(alphabet_size_ - 1, writer);

      return true;
    }

    /// Small tree.
    if (num_symbols_ <= kMaxNumSymbolsForSmallCode) {
      // Small tree marker to encode 1-2 symbols.
      writer->Write(1, 1);
      if (num_symbols_ == 0) {
        writer->Write(1, 0);
        StoreVarLenUint8(0, writer);
      } else {
        writer->Write(1, num_symbols_ - 1);
        for (size_t i = 0; i < num_symbols_; ++i) {
          StoreVarLenUint8(symbols_[i], writer);
        }
      }
      if (num_symbols_ == 2) {
        writer->Write(ANS_LOG_TAB_SIZE, counts_[symbols_[0]]);
      }

      return true;
    }

    /// General tree.
    // Mark non-small tree.
    writer->Write(1, 0);
    // Mark non-flat histogram.
    writer->Write(1, 0);

    // Elias gamma-like code for `shift = method - 1`. Only difference is that
    // if the number of bits to be encoded is equal to `upper_bound_log`,
    // we skip the terminating 0 in unary coding.
    int upper_bound_log = FloorLog2Nonzero(ANS_LOG_TAB_SIZE + 1);
    int log = FloorLog2Nonzero(method_);
    writer->Write(log, (1 << log) - 1);
    if (log != upper_bound_log) writer->Write(1, 0);
    writer->Write(log, ((1 << log) - 1) & method_);

    // Since `num_symbols_ >= 3`, we know that `alphabet_size_ >= 3`, therefore
    // we encode `alphabet_size_ - 3`.
    StoreVarLenUint8(alphabet_size_ - 3, writer);

    // Precompute sequences for RLE encoding. Contains the number of identical
    // values starting at a given index. Only contains that value at the first
    // element of the series.
    uint8_t same[ANS_MAX_ALPHABET_SIZE] = {};
    size_t last = 0;
    for (size_t i = 1; i <= alphabet_size_; i++) {
      // Store the sequence length once different symbol reached, or we are
      // near the omit_pos_, or we're at the end. We don't support including the
      // omit_pos_ in an RLE sequence because this value may use a different
      // amount of log2 bits than standard, it is too complex to handle in the
      // decoder.
      if (i == alphabet_size_ || i == omit_pos_ || i == omit_pos_ + 1 ||
          counts_[i] != counts_[last]) {
        same[last] = i - last;
        last = i;
      }
    }

    uint8_t bit_width[ANS_MAX_ALPHABET_SIZE] = {};
    // Use shortest possible Huffman code to encode `omit_pos` (see
    // `kBitWidthLengths`). `bit_width` value at `omit_pos` should be the
    // first of maximal values in the whole `bit_width` array, so it can be
    // increased without changing that property
    int omit_width = 10;
    for (size_t i = 0; i < alphabet_size_; ++i) {
      if (i != omit_pos_ && counts_[i] > 0) {
        bit_width[i] = FloorLog2Nonzero<uint32_t>(counts_[i]) + 1;
        omit_width = std::max(omit_width, bit_width[i] + int{i < omit_pos_});
      }
    }
    bit_width[omit_pos_] = static_cast<uint8_t>(omit_width);

    // The bit widths are encoded with a static Huffman code.
    // The last symbol is used as RLE sequence.
    constexpr uint8_t kBitWidthLengths[ANS_LOG_TAB_SIZE + 2] = {
        5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7,
    };
    constexpr uint8_t kBitWidthSymbols[ANS_LOG_TAB_SIZE + 2] = {
        17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65,
    };
    constexpr uint8_t kMinReps = 5;
    constexpr size_t rep = ANS_LOG_TAB_SIZE + 1;
    // Encode count bit widths
    for (size_t i = 0; i < alphabet_size_; ++i) {
      writer->Write(kBitWidthLengths[bit_width[i]],
                    kBitWidthSymbols[bit_width[i]]);
      if (same[i] >= kMinReps) {
        // Encode the RLE symbol and skip the repeated ones.
        writer->Write(kBitWidthLengths[rep], kBitWidthSymbols[rep]);
        StoreVarLenUint8(same[i] - kMinReps, writer);
        i += same[i] - 1;
      }
    }
    // Encode additional bits of accuracy
    uint32_t shift = method_ - 1;
    if (shift != 0) {  // otherwise `bitcount = 0`
      for (size_t i = 0; i < alphabet_size_; ++i) {
        if (bit_width[i] > 1 && i != omit_pos_) {
          int bitcount = GetPopulationCountPrecision(bit_width[i] - 1, shift);
          int drop_bits = bit_width[i] - 1 - bitcount;
          JXL_DASSERT((counts_[i] & ((1 << drop_bits) - 1)) == 0);
          writer->Write(bitcount, (counts_[i] >> drop_bits) - (1 << bitcount));
        }
        if (same[i] >= kMinReps) {
          // Skip symbols encoded by RLE.
          i += same[i] - 1;
        }
      }
    }
    return true;
  }

  void ANSBuildInfoTable(const AliasTable::Entry* table, size_t log_alpha_size,
                         ANSEncSymbolInfo* info) {
    // Create valid alias table for empty streams
    for (size_t s = 0; s < std::max(size_t{1}, alphabet_size_); ++s) {
      const ANSHistBin freq = s == alphabet_size_ ? ANS_TAB_SIZE : counts_[s];
      info[s].freq_ = static_cast<uint16_t>(freq);
#ifdef USE_MULT_BY_RECIPROCAL
      if (freq != 0) {
        info[s].ifreq_ = ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) /
                         info[s].freq_;
      } else {
        info[s].ifreq_ =
            1;  // Shouldn't matter (symbol shouldn't occur), but...
      }
#endif
      info[s].reverse_map_.resize(freq);
    }
    size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
    size_t entry_size_minus_1 = (1 << log_entry_size) - 1;
    for (int i = 0; i < ANS_TAB_SIZE; i++) {
      AliasTable::Symbol s =
          AliasTable::Lookup(table, i, log_entry_size, entry_size_minus_1);
      info[s.value].reverse_map_[s.offset] = i;
    }
  }

 private:
  ANSEncodingHistogram() {}

  // Fixed-point log2 LUT for values of [0,4096]
  using Lg2LUT = std::array<uint32_t, ANS_TAB_SIZE + 1>;
  static const Lg2LUT lg2;

  float EstimateDataBits(const Histogram& histo) {
    int64_t sum = 0;
    for (size_t i = 0; i < alphabet_size_; ++i) {
      // += histogram[i] * -log(counts[i]/total_counts)
      sum += histo.counts[i] * int64_t{lg2[counts_[i]]};
    }
    return (histo.total_count - ldexpf(sum, -31)) * ANS_LOG_TAB_SIZE;
  }

  static float EstimateDataBitsFlat(const Histogram& histo) {
    size_t len = histo.alphabet_size();
    int64_t flat_bits = int64_t{lg2[len]} * ANS_LOG_TAB_SIZE;
    return ldexpf(histo.total_count * flat_bits, -31);
  }

  struct CountsEntropy {
    ANSHistBin count : 16;     // allowed value of counts in a histogram bin
    ANSHistBin step_log : 16;  // log2 of increase step size (can use 5 bits)
    int32_t delta_lg2;  // change of log between that value and the next allowed
  };

  // Array is sorted by decreasing allowed counts for each possible shift.
  // Exclusion of single-bin histograms before `RebalanceHistogram` allows
  // to put count upper limit of 4095, and shifts of 11 and 12 produce the
  // same table
  using CountsArray =
      std::array<std::array<CountsEntropy, ANS_TAB_SIZE>, ANS_LOG_TAB_SIZE>;
  using CountsIndex =
      std::array<std::array<uint16_t, ANS_TAB_SIZE>, ANS_LOG_TAB_SIZE>;
  struct AllowedCounts {
    CountsArray array;
    CountsIndex index;
  };
  static const AllowedCounts allowed_counts;

  // Returns the difference between largest count that can be represented and is
  // smaller than "count" and smallest representable count larger than "count".
  static uint32_t SmallestIncrementLog(uint32_t count, uint32_t shift) {
    if (count == 0) return 0;
    uint32_t bits = FloorLog2Nonzero(count);
    uint32_t drop_bits = bits - GetPopulationCountPrecision(bits, shift);
    return drop_bits;
  }
  // We are growing/reducing histogram step by step trying to maximize total
  // entropy i.e. sum of `freq[n] * log[counts[n]]` with a given sum of
  // `counts[n]` chosen from `allowed_counts[shift]`. This sum is balanced by
  // the `counts[omit_pos_]` in the highest bin of histogram. We start from
  // close to correct solution and each time a step with maximum entropy
  // increase per unit of bin change is chosen. This greedy scheme is not
  // guaranteed to achieve the global maximum, but cannot produce invalid
  // histogram. We use a fixed-point approximation for logarithms and all
  // arithmetic is integer besides initial approximation. Sum of `freq` and each
  // of `lg2[counts]` are supposed to be limited to `int32_t` range, so that the
  // sum of their products should not exceed `int64_t`.
  bool RebalanceHistogram(const Histogram& histo) {
    constexpr ANSHistBin table_size = ANS_TAB_SIZE;
    uint32_t shift = method_ - 1;

    struct EntropyDelta {
      ANSHistBin freq;   // initial count
      size_t count_ind;  // index of current bin value in `allowed_counts`
      size_t bin_ind;    // index of current bin in `counts`
    };
    // Penalties corresponding to different step sizes - entropy decrease in
    // balancing bin, step of size (1 << ANS_LOG_TAB_SIZE - 1) is not possible
    std::array<int64_t, ANS_LOG_TAB_SIZE - 1> balance_inc = {};
    std::array<int64_t, ANS_LOG_TAB_SIZE - 1> balance_dec = {};
    const auto& ac = allowed_counts.array[shift];
    const auto& ai = allowed_counts.index[shift];
    // TODO(ivan) separate cases of shift >= 11 - all steps are 1 there, and
    // possibly 10 - all relevant steps are 2.
    // Total entropy change by a step: increase/decrease in current bin
    // together with corresponding decrease/increase in the balancing bin.
    // Inc steps increase current bin, dec steps decrease
    const auto delta_entropy_inc = [&](const EntropyDelta& a) {
      return a.freq * int64_t{ac[a.count_ind].delta_lg2} -
             balance_inc[ac[a.count_ind].step_log];
    };
    const auto delta_entropy_dec = [&](const EntropyDelta& a) {
      return a.freq * int64_t{ac[a.count_ind + 1].delta_lg2} -
             balance_dec[ac[a.count_ind + 1].step_log];
    };
    // Compare steps by entropy increase per unit of histogram bin change.
    // Truncation is OK here, accuracy is anyway better than float
    const auto IncLess = [&](const EntropyDelta& a, const EntropyDelta& b) {
      return delta_entropy_inc(a) >> ac[a.count_ind].step_log <
             delta_entropy_inc(b) >> ac[b.count_ind].step_log;
    };
    const auto DecLess = [&](const EntropyDelta& a, const EntropyDelta& b) {
      return delta_entropy_dec(a) >> ac[a.count_ind + 1].step_log <
             delta_entropy_dec(b) >> ac[b.count_ind + 1].step_log;
    };
    // Vector of adjustable bins from `allowed_counts`
    std::vector<EntropyDelta> bins;
    bins.reserve(256);

    double norm = double{table_size} / histo.total_count;

    size_t remainder_pos = 0;  // highest balancing bin in the histogram
    int64_t max_freq = 0;
    ANSHistBin rest = table_size;  // reserve of histogram counts to distribute
    for (size_t n = 0; n < alphabet_size_; ++n) {
      ANSHistBin freq = histo.counts[n];
      if (freq > max_freq) {
        remainder_pos = n;
        max_freq = freq;
      }

      double target = freq * norm;  // rounding
      // Keep zeros and clamp nonzero freq counts to [1, table_size)
      ANSHistBin count = std::max<ANSHistBin>(round(target), freq > 0);
      count = std::min<ANSHistBin>(count, table_size - 1);
      uint32_t step_log = SmallestIncrementLog(count, shift);
      ANSHistBin inc = 1 << step_log;
      count &= ~(inc - 1);

      counts_[n] = count;
      rest -= count;
      if (target > 1.0) {
        bins.push_back({freq, ai[count], n});
      }
    }

    // Delete the highest balancing bin from adjustable by `allowed_counts`
    bins.erase(std::find_if(
        bins.begin(), bins.end(),
        [&](const EntropyDelta& a) { return a.bin_ind == remainder_pos; }));
    // From now on `rest` is the height of balancing bin,
    // here it can be negative, but will be tracted into positive domain later
    rest += counts_[remainder_pos];

    if (!bins.empty()) {
      const uint32_t max_log = ac[1].step_log;
      while (true) {
        // Update balancing bin penalties setting guards and tractors
        for (uint32_t log = 0; log <= max_log; ++log) {
          ANSHistBin delta = 1 << log;
          if (rest >= table_size) {
            // Tract large `rest` into allowed domain:
            balance_inc[log] = 0;  // permit all inc steps
            balance_dec[log] = 0;  // forbid all dec steps
          } else if (rest > 1) {
            // `rest` is OK, put guards against non-possible steps
            balance_inc[log] =
                rest > delta  // possible step
                    ? max_freq * int64_t{lg2[rest] - lg2[rest - delta]}
                    : std::numeric_limits<int64_t>::max();  // forbidden
            balance_dec[log] =
                rest + delta < table_size  // possible step
                    ? max_freq * int64_t{lg2[rest + delta] - lg2[rest]}
                    : 0;  // forbidden
          } else {
            // Tract negative or zero `rest` into positive:
            // forbid all inc steps
            balance_inc[log] = std::numeric_limits<int64_t>::max();
            // permit all dec steps
            balance_dec[log] = std::numeric_limits<int64_t>::max();
          }
        }
        // Try to increase entropy
        auto best_bin_inc = std::max_element(bins.begin(), bins.end(), IncLess);
        if (delta_entropy_inc(*best_bin_inc) > 0) {
          // Grow the bin with the best histogram entropy increase
          rest -= 1 << ac[best_bin_inc->count_ind--].step_log;
        } else {
          // This still implies that entropy is strictly increasing each step
          // (or `rest` is tracted into positive domain), so we cannot loop
          // infinitely
          auto best_bin_dec =
              std::min_element(bins.begin(), bins.end(), DecLess);
          // Break if no reverse steps can grow entropy (or valid)
          if (delta_entropy_dec(*best_bin_dec) >= 0) break;
          // Decrease the bin with the best histogram entropy increase
          rest += 1 << ac[++best_bin_dec->count_ind].step_log;
        }
      }
      // Set counts besides the balancing bin
      for (auto& a : bins) counts_[a.bin_ind] = ac[a.count_ind].count;

      // The scheme works fine if we have room to grow `bit_width` of balancing
      // bin, otherwise we need to put balancing bin to the first bin of 12 bit
      // width. In this case both that bin and balancing one should be close to
      // 2048 in targets, so exchange of them will not produce much worse
      // histogram
      for (size_t n = 0; n < remainder_pos; ++n) {
        if (counts_[n] >= 2048) {
          counts_[remainder_pos] = counts_[n];
          remainder_pos = n;
          break;
        }
      }
    }
    // Set balancing bin
    counts_[remainder_pos] = rest;
    omit_pos_ = remainder_pos;

    return counts_[remainder_pos] > 0;
  }

  float cost_ = 0;
  uint32_t method_ = 0;
  size_t omit_pos_ = 0;
  size_t alphabet_size_ = 0;
  size_t num_symbols_ = 0;
  size_t symbols_[kMaxNumSymbolsForSmallCode] = {};
  std::vector<ANSHistBin> counts_{};
};

using AEH = ANSEncodingHistogram;

const AEH::Lg2LUT AEH::lg2 = [] {
  Lg2LUT lg2;
  lg2[0] = 0;  // for entropy calculations it is OK
  for (size_t i = 1; i < lg2.size(); ++i) {
    lg2[i] = round(ldexp(log2(i) / ANS_LOG_TAB_SIZE, 31));
  }
  return lg2;
}();

const AEH::AllowedCounts AEH::allowed_counts = [] {
  AllowedCounts result;

  for (uint32_t shift = 0; shift < result.array.size(); ++shift) {
    auto& ac = result.array[shift];
    auto& ai = result.index[shift];
    ANSHistBin last = ~0;
    size_t slot = 0;
    // TODO(eustas): are those "default" values relevant?
    ac[0].delta_lg2 = 0;
    ac[0].step_log = 0;
    for (int32_t i = ac.size() - 1; i >= 0; --i) {
      int32_t curr = i & ~((1 << SmallestIncrementLog(i, shift)) - 1);
      if (curr == last) continue;
      last = curr;
      ac[slot].count = curr;
      ai[curr] = slot;
      if (curr == 0) {
        // Guards against non-possible steps:
        // at max value [0] - 0 (by init), at min value - max
        ac[slot].delta_lg2 = std::numeric_limits<int32_t>::max();
        ac[slot].step_log = 0;
      } else if (slot > 0) {
        ANSHistBin prev = ac[slot - 1].count;
        ac[slot].delta_lg2 = round(ldexp(
            log2(static_cast<double>(prev) / curr) / ANS_LOG_TAB_SIZE, 31));
        ac[slot].step_log = FloorLog2Nonzero<uint32_t>(prev - curr);
        prev = curr;
      }
      slot++;
    }
  }

  return result;
}();

}  // namespace

StatusOr<float> Histogram::ANSPopulationCost() const {
  if (counts.size() > ANS_MAX_ALPHABET_SIZE) {
    return std::numeric_limits<float>::max();
  }
  JXL_ASSIGN_OR_RETURN(
      ANSEncodingHistogram normalized,
      ANSEncodingHistogram::ComputeBest(
          *this, HistogramParams::ANSHistogramStrategy::kFast));
  return normalized.Cost();
}

// Returns an estimate or exact cost of encoding this histogram and the
// corresponding data.
StatusOr<size_t> EntropyEncodingData::BuildAndStoreANSEncodingData(
    JxlMemoryManager* memory_manager,
    HistogramParams::ANSHistogramStrategy ans_histogram_strategy,
    const Histogram& histogram, BitWriter* writer) {
  ANSEncSymbolInfo* info = encoding_info.back().data();
  size_t size = histogram.alphabet_size();
  if (use_prefix_code) {
    size_t cost = 0;
    if (size <= 1) return 0;
    std::vector<uint32_t> histo(size);
    for (size_t i = 0; i < size; i++) {
      JXL_ENSURE(histogram.counts[i] >= 0);
      histo[i] = histogram.counts[i];
    }
    std::vector<uint8_t> depths(size);
    std::vector<uint16_t> bits(size);
    if (writer == nullptr) {
      BitWriter tmp_writer{memory_manager};
      JXL_RETURN_IF_ERROR(tmp_writer.WithMaxBits(
          8 * size + 8,  // safe upper bound
          LayerType::Header, /*aux_out=*/nullptr, [&] {
            return BuildAndStoreHuffmanTree(histo.data(), size, depths.data(),
                                            bits.data(), &tmp_writer);
          }));
      cost = tmp_writer.BitsWritten();
    } else {
      size_t start = writer->BitsWritten();
      JXL_RETURN_IF_ERROR(BuildAndStoreHuffmanTree(
          histo.data(), size, depths.data(), bits.data(), writer));
      cost = writer->BitsWritten() - start;
    }
    for (size_t i = 0; i < size; i++) {
      info[i].bits = depths[i] == 0 ? 0 : bits[i];
      info[i].depth = depths[i];
    }
    // Estimate data cost.
    for (size_t i = 0; i < size; i++) {
      cost += histo[i] * info[i].depth;
    }
    return cost;
  }
  JXL_ASSIGN_OR_RETURN(
      ANSEncodingHistogram normalized,
      ANSEncodingHistogram::ComputeBest(histogram, ans_histogram_strategy));

  // TODO(eustas): fix: 2KiB on stack
  AliasTable::Entry a[ANS_MAX_ALPHABET_SIZE];

  JXL_RETURN_IF_ERROR(
      InitAliasTable(normalized.Counts(), ANS_LOG_TAB_SIZE, log_alpha_size, a));
  normalized.ANSBuildInfoTable(a, log_alpha_size, info);
  if (writer != nullptr) {
    // size_t start = writer->BitsWritten();
    JXL_RETURN_IF_ERROR(normalized.Encode(writer));
    // return writer->BitsWritten() - start;
  }
  return static_cast<size_t>(ceilf(normalized.Cost()));
}

namespace {

Histogram HistogramFromSymbolInfo(
    const std::vector<ANSEncSymbolInfo>& encoding_info, bool use_prefix_code) {
  Histogram histo;
  histo.counts.resize(DivCeil(encoding_info.size(), Histogram::kRounding) *
                      Histogram::kRounding);
  histo.total_count = 0;
  for (size_t i = 0; i < encoding_info.size(); ++i) {
    const ANSEncSymbolInfo& info = encoding_info[i];
    int count = use_prefix_code
                    ? (info.depth ? (1u << (PREFIX_MAX_BITS - info.depth)) : 0)
                    : info.freq_;
    histo.counts[i] = count;
    histo.total_count += count;
  }
  return histo;
}

}  // namespace

Status EntropyEncodingData::ChooseUintConfigs(
    JxlMemoryManager* memory_manager, const HistogramParams& params,
    const std::vector<std::vector<Token>>& tokens,
    std::vector<Histogram>& clustered_histograms) {
  // Set sane default `log_alpha_size`.
  if (use_prefix_code) {
    log_alpha_size = PREFIX_MAX_BITS;
  } else if (params.streaming_mode) {
    // TODO(szabadka) Figure out if we can use lower values here.
    log_alpha_size = 8;
  } else if (lz77.enabled) {
    log_alpha_size = 8;
  } else {
    log_alpha_size = 7;
  }

  if (ans_fuzzer_friendly_) {
    uint_config.assign(1, HybridUintConfig(7, 0, 0));
    return true;
  }

  uint_config.assign(clustered_histograms.size(), params.UintConfig());
  // If the uint config is fixed, just use it.
  if (params.uint_method != HistogramParams::HybridUintMethod::kBest &&
      params.uint_method != HistogramParams::HybridUintMethod::kFast) {
    return true;
  }
  // Even if the uint config is adaptive, just stick with the default in
  // streaming mode.
  if (params.streaming_mode) {
    return true;
  }

  // Brute-force method that tries a few options.
  std::vector<HybridUintConfig> configs;
  if (params.uint_method == HistogramParams::HybridUintMethod::kBest) {
    configs = {
        HybridUintConfig(4, 2, 0),  // default
        HybridUintConfig(4, 1, 0),  // less precise
        HybridUintConfig(4, 2, 1),  // add sign
        HybridUintConfig(4, 2, 2),  // add sign+parity
        HybridUintConfig(4, 1, 2),  // add parity but less msb
        // Same as above, but more direct coding.
        HybridUintConfig(5, 2, 0), HybridUintConfig(5, 1, 0),
        HybridUintConfig(5, 2, 1), HybridUintConfig(5, 2, 2),
        HybridUintConfig(5, 1, 2),
        // Same as above, but less direct coding.
        HybridUintConfig(3, 2, 0), HybridUintConfig(3, 1, 0),
        HybridUintConfig(3, 2, 1), HybridUintConfig(3, 1, 2),
        // For near-lossless.
        HybridUintConfig(4, 1, 3), HybridUintConfig(5, 1, 4),
        HybridUintConfig(5, 2, 3), HybridUintConfig(6, 1, 5),
        HybridUintConfig(6, 2, 4), HybridUintConfig(6, 0, 0),
        // Other
        HybridUintConfig(0, 0, 0),   // varlenuint
        HybridUintConfig(2, 0, 1),   // works well for ctx map
        HybridUintConfig(7, 0, 0),   // direct coding
        HybridUintConfig(8, 0, 0),   // direct coding
        HybridUintConfig(9, 0, 0),   // direct coding
        HybridUintConfig(10, 0, 0),  // direct coding
        HybridUintConfig(11, 0, 0),  // direct coding
        HybridUintConfig(12, 0, 0),  // direct coding
    };
  } else {
    JXL_DASSERT(params.uint_method == HistogramParams::HybridUintMethod::kFast);
    configs = {
        HybridUintConfig(4, 2, 0),  // default
        HybridUintConfig(4, 1, 2),  // add parity but less msb
        HybridUintConfig(0, 0, 0),  // smallest histograms
        HybridUintConfig(2, 0, 1),  // works well for ctx map
    };
  }

  size_t num_histo = clustered_histograms.size();
  std::vector<uint8_t> is_valid(num_histo);
  std::vector<size_t> histo_volume(2 * num_histo);
  std::vector<size_t> histo_offset(2 * num_histo + 1);
  std::vector<uint32_t> max_value_per_histo(2 * num_histo);

  // TODO(veluca): do not ignore lz77 commands.

  for (const auto& stream : tokens) {
    for (const auto& token : stream) {
      size_t histo = context_map[token.context];
      histo_volume[histo + (token.is_lz77_length ? num_histo : 0)]++;
    }
  }
  size_t max_histo_volume = 0;
  for (size_t h = 0; h < 2 * num_histo; ++h) {
    max_histo_volume = std::max(max_histo_volume, histo_volume[h]);
    histo_offset[h + 1] = histo_offset[h] + histo_volume[h];
  }

  const size_t max_vec_size = MaxVectorSize();
  std::vector<uint32_t> transposed(histo_offset[num_histo * 2] + max_vec_size);
  {
    std::vector<size_t> next_offset = histo_offset;  // copy
    for (const auto& stream : tokens) {
      for (const auto& token : stream) {
        size_t histo =
            context_map[token.context] + (token.is_lz77_length ? num_histo : 0);
        transposed[next_offset[histo]++] = token.value;
      }
    }
  }
  for (size_t h = 0; h < 2 * num_histo; ++h) {
    max_value_per_histo[h] =
        MaxValue(transposed.data() + histo_offset[h], histo_volume[h]);
  }
  uint32_t max_lz77 = 0;
  for (size_t h = num_histo; h < 2 * num_histo; ++h) {
    max_lz77 = std::max(max_lz77, MaxValue(transposed.data() + histo_offset[h],
                                           histo_volume[h]));
  }

  // Wider histograms are assigned max cost in PopulationCost anyway
  // and therefore will not be used
  size_t max_alpha = ANS_MAX_ALPHABET_SIZE;

  JXL_ASSIGN_OR_RETURN(
      AlignedMemory tmp,
      AlignedMemory::Create(memory_manager, (max_histo_volume + max_vec_size) *
                                                sizeof(uint32_t)));
  for (size_t h = 0; h < num_histo; h++) {
    float best_cost = std::numeric_limits<float>::max();
    for (HybridUintConfig cfg : configs) {
      uint32_t max_v = max_value_per_histo[h];
      size_t capacity;
      {
        uint32_t tok, nbits, bits;
        cfg.Encode(max_v, &tok, &nbits, &bits);
        tok |= cfg.LsbMask();
        if (tok >= max_alpha || (lz77.enabled && tok >= lz77.min_symbol)) {
          continue;  // Not valid config for this context
        }
        capacity = tok + 1;
      }

      Histogram histo;
      histo.EnsureCapacity(capacity);
      size_t len = histo_volume[h];
      uint32_t* data = transposed.data() + histo_offset[h];
      size_t extra_bits = EstimateTokenCost(data, len, cfg, tmp);
      uint32_t* tmp_tokens = tmp.address<uint32_t>();
      for (size_t i = 0; i < len; ++i) {
        histo.FastAdd(tmp_tokens[i]);
      }
      histo.Condition();
      JXL_ASSIGN_OR_RETURN(float cost, histo.ANSPopulationCost());
      cost += extra_bits;
      // Add signaling cost of the hybriduintconfig itself.
      cost += CeilLog2Nonzero(cfg.split_exponent + 1);
      cost += CeilLog2Nonzero(cfg.split_exponent - cfg.msb_in_token + 1);
      if (cost < best_cost) {
        uint_config[h] = cfg;
        best_cost = cost;
        clustered_histograms[h].swap(histo);
      }
    }
  }

  size_t max_tok = 0;
  for (size_t h = 0; h < num_histo; ++h) {
    Histogram& histo = clustered_histograms[h];
    max_tok = std::max(max_tok, histo.MaxSymbol());
    size_t len = histo_volume[num_histo + h];
    if (len == 0) continue;  // E.g. when lz77 not enabled
    size_t max_histo_tok = max_value_per_histo[num_histo + h];
    uint32_t tok, nbits, bits;
    lz77.length_uint_config.Encode(max_histo_tok, &tok, &nbits, &bits);
    tok |= lz77.length_uint_config.LsbMask();
    tok += lz77.min_symbol;
    histo.EnsureCapacity(tok + 1);
    uint32_t* data = transposed.data() + histo_offset[num_histo + h];
    uint32_t unused =
        EstimateTokenCost(data, len, lz77.length_uint_config, tmp);
    (void)unused;
    uint32_t* tmp_tokens = tmp.address<uint32_t>();
    for (size_t i = 0; i < len; ++i) {
      histo.FastAdd(tmp_tokens[i] + lz77.min_symbol);
    }
    histo.Condition();
    max_tok = std::max(max_tok, histo.MaxSymbol());
  }

  // `log_alpha_size - 5` is encoded in the header, so min is 5.
  size_t log_size = 5;
  while (max_tok >= (1u << log_size)) ++log_size;

  size_t max_log_alpha_size = use_prefix_code ? PREFIX_MAX_BITS : 8;
  JXL_ENSURE(log_size <= max_log_alpha_size);

  if (use_prefix_code) {
    log_alpha_size = PREFIX_MAX_BITS;
  } else {
    log_alpha_size = log_size;
  }

  return true;
}

// NOTE: `layer` is only for clustered_entropy; caller does ReclaimAndCharge.
// Returns cost (in bits).
StatusOr<size_t> EntropyEncodingData::BuildAndStoreEntropyCodes(
    JxlMemoryManager* memory_manager, const HistogramParams& params,
    const std::vector<std::vector<Token>>& tokens,
    const std::vector<Histogram>& builder, BitWriter* writer, LayerType layer,
    AuxOut* aux_out) {
  const size_t prev_histograms = encoding_info.size();
  std::vector<Histogram> clustered_histograms;
  for (size_t i = 0; i < prev_histograms; ++i) {
    clustered_histograms.push_back(
        HistogramFromSymbolInfo(encoding_info[i], use_prefix_code));
  }
  size_t context_offset = context_map.size();
  context_map.resize(context_offset + builder.size());
  if (builder.size() > 1) {
    if (!ans_fuzzer_friendly_) {
      std::vector<uint32_t> histogram_symbols;
      JXL_RETURN_IF_ERROR(ClusterHistograms(params, builder, kClustersLimit,
                                            &clustered_histograms,
                                            &histogram_symbols));
      for (size_t c = 0; c < builder.size(); ++c) {
        context_map[context_offset + c] =
            static_cast<uint8_t>(histogram_symbols[c]);
      }
    } else {
      JXL_ENSURE(encoding_info.empty());
      std::fill(context_map.begin(), context_map.end(), 0);
      size_t max_symbol = 0;
      for (const Histogram& h : builder) {
        max_symbol = std::max(h.counts.size(), max_symbol);
      }
      size_t num_symbols = 1 << CeilLog2Nonzero(max_symbol + 1);
      clustered_histograms.resize(1);
      clustered_histograms[0].Clear();
      for (size_t i = 0; i < num_symbols; i++) {
        clustered_histograms[0].Add(i);
      }
    }
    if (writer != nullptr) {
      JXL_RETURN_IF_ERROR(EncodeContextMap(
          context_map, clustered_histograms.size(), writer, layer, aux_out));
    }
  } else {
    JXL_ENSURE(encoding_info.empty());
    clustered_histograms.push_back(builder[0]);
  }
  if (aux_out != nullptr) {
    for (size_t i = prev_histograms; i < clustered_histograms.size(); ++i) {
      aux_out->layer(layer).clustered_entropy +=
          clustered_histograms[i].ShannonEntropy();
    }
  }

  JXL_RETURN_IF_ERROR(
      ChooseUintConfigs(memory_manager, params, tokens, clustered_histograms));

  SizeWriter size_writer;  // Used if writer == nullptr to estimate costs.
  size_t cost = use_prefix_code ? 1 : 3;

  if (writer) writer->Write(1, TO_JXL_BOOL(use_prefix_code));
  if (writer == nullptr) {
    EncodeUintConfigs(uint_config, &size_writer, log_alpha_size);
  } else {
    if (!use_prefix_code) writer->Write(2, log_alpha_size - 5);
    EncodeUintConfigs(uint_config, writer, log_alpha_size);
  }
  if (use_prefix_code) {
    for (const auto& histo : clustered_histograms) {
      size_t alphabet_size = std::max<size_t>(1, histo.alphabet_size());
      if (writer) {
        StoreVarLenUint16(alphabet_size - 1, writer);
      } else {
        StoreVarLenUint16(alphabet_size - 1, &size_writer);
      }
    }
  }
  cost += size_writer.size;
  for (size_t c = prev_histograms; c < clustered_histograms.size(); ++c) {
    size_t alphabet_size = clustered_histograms[c].alphabet_size();
    encoding_info.emplace_back();
    encoding_info.back().resize(alphabet_size);
    BitWriter* histo_writer = writer;
    if (params.streaming_mode) {
      encoded_histograms.emplace_back(memory_manager);
      histo_writer = &encoded_histograms.back();
    }
    const auto& body = [&]() -> Status {
      JXL_ASSIGN_OR_RETURN(size_t ans_cost,
                           BuildAndStoreANSEncodingData(
                               memory_manager, params.ans_histogram_strategy,
                               clustered_histograms[c], histo_writer));
      cost += ans_cost;
      return true;
    };
    if (histo_writer) {
      JXL_RETURN_IF_ERROR(histo_writer->WithMaxBits(
          256 + alphabet_size * 24, layer, aux_out, body,
          /*finished_histogram=*/true));
    } else {
      JXL_RETURN_IF_ERROR(body());
    }
    if (params.streaming_mode) {
      JXL_RETURN_IF_ERROR(writer->AppendUnaligned(*histo_writer));
    }
  }
  return cost;
}

template <typename Writer>
void EncodeUintConfig(const HybridUintConfig uint_config, Writer* writer,
                      size_t log_alpha_size) {
  writer->Write(CeilLog2Nonzero(log_alpha_size + 1),
                uint_config.split_exponent);
  if (uint_config.split_exponent == log_alpha_size) {
    return;  // msb/lsb don't matter.
  }
  size_t nbits = CeilLog2Nonzero(uint_config.split_exponent + 1);
  writer->Write(nbits, uint_config.msb_in_token);
  nbits = CeilLog2Nonzero(uint_config.split_exponent -
                          uint_config.msb_in_token + 1);
  writer->Write(nbits, uint_config.lsb_in_token);
}
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size) {
  // TODO(veluca): RLE?
  for (const auto& cfg : uint_config) {
    EncodeUintConfig(cfg, writer, log_alpha_size);
  }
}
template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                BitWriter*, size_t);

Status EncodeHistograms(const EntropyEncodingData& codes, BitWriter* writer,
                        LayerType layer, AuxOut* aux_out) {
  return writer->WithMaxBits(
      128 + kClustersLimit * 136, layer, aux_out,
      [&]() -> Status {
        JXL_RETURN_IF_ERROR(Bundle::Write(codes.lz77, writer, layer, aux_out));
        if (codes.lz77.enabled) {
          EncodeUintConfig(codes.lz77.length_uint_config, writer,
                           /*log_alpha_size=*/8);
        }
        JXL_RETURN_IF_ERROR(EncodeContextMap(codes.context_map,
                                             codes.encoding_info.size(), writer,
                                             layer, aux_out));
        writer->Write(1, TO_JXL_BOOL(codes.use_prefix_code));
        size_t log_alpha_size = 8;
        if (codes.use_prefix_code) {
          log_alpha_size = PREFIX_MAX_BITS;
        } else {
          log_alpha_size = 8;  // streaming_mode
          writer->Write(2, log_alpha_size - 5);
        }
        EncodeUintConfigs(codes.uint_config, writer, log_alpha_size);
        if (codes.use_prefix_code) {
          for (const auto& info : codes.encoding_info) {
            StoreVarLenUint16(info.size() - 1, writer);
          }
        }
        for (const auto& histo_writer : codes.encoded_histograms) {
          JXL_RETURN_IF_ERROR(writer->AppendUnaligned(histo_writer));
        }
        return true;
      },
      /*finished_histogram=*/true);
}

StatusOr<size_t> BuildAndEncodeHistograms(
    JxlMemoryManager* memory_manager, const HistogramParams& params,
    size_t num_contexts, std::vector<std::vector<Token>>& tokens,
    EntropyEncodingData* codes, BitWriter* writer, LayerType layer,
    AuxOut* aux_out) {
  // TODO(Ivan): presumably not needed - default
  // if (params.initialize_global_state) codes->lz77.enabled = false;
  codes->lz77.nonserialized_distance_context = num_contexts;
  codes->lz77.min_symbol = params.force_huffman ? 512 : 224;
  std::vector<std::vector<Token>> tokens_lz77 =
      ApplyLZ77(params, num_contexts, tokens, codes->lz77);
  if (!tokens_lz77.empty()) codes->lz77.enabled = true;
  if (ans_fuzzer_friendly_) {
    codes->lz77.length_uint_config = HybridUintConfig(10, 0, 0);
    codes->lz77.min_symbol = 2048;
  }

  size_t cost = 0;
  const size_t max_contexts = std::min(num_contexts, kClustersLimit);
  const auto& body = [&]() -> Status {
    if (writer) {
      JXL_RETURN_IF_ERROR(Bundle::Write(codes->lz77, writer, layer, aux_out));
    } else {
      size_t ebits, bits;
      JXL_RETURN_IF_ERROR(Bundle::CanEncode(codes->lz77, &ebits, &bits));
      cost += bits;
    }
    if (codes->lz77.enabled) {
      if (writer) {
        size_t b = writer->BitsWritten();
        EncodeUintConfig(codes->lz77.length_uint_config, writer,
                         /*log_alpha_size=*/8);
        cost += writer->BitsWritten() - b;
      } else {
        SizeWriter size_writer;
        EncodeUintConfig(codes->lz77.length_uint_config, &size_writer,
                         /*log_alpha_size=*/8);
        cost += size_writer.size;
      }
      num_contexts += 1;
      tokens = std::move(tokens_lz77);
    }
    size_t total_tokens = 0;
    // Build histograms.
    std::vector<Histogram> builder(num_contexts);
    HybridUintConfig uint_config = params.UintConfig();
    if (ans_fuzzer_friendly_) {
      uint_config = HybridUintConfig(10, 0, 0);
    }
    for (const auto& stream : tokens) {
      if (codes->lz77.enabled) {
        for (const auto& token : stream) {
          total_tokens++;
          uint32_t tok, nbits, bits;
          (token.is_lz77_length ? codes->lz77.length_uint_config : uint_config)
              .Encode(token.value, &tok, &nbits, &bits);
          tok += token.is_lz77_length ? codes->lz77.min_symbol : 0;
          JXL_DASSERT(token.context < num_contexts);
          builder[token.context].Add(tok);
        }
      } else if (num_contexts == 1) {
        for (const auto& token : stream) {
          total_tokens++;
          uint32_t tok, nbits, bits;
          uint_config.Encode(token.value, &tok, &nbits, &bits);
          builder[0].Add(tok);
        }
      } else {
        for (const auto& token : stream) {
          total_tokens++;
          uint32_t tok, nbits, bits;
          uint_config.Encode(token.value, &tok, &nbits, &bits);
          JXL_DASSERT(token.context < num_contexts);
          builder[token.context].Add(tok);
        }
      }
    }

    if (params.add_missing_symbols) {
      for (size_t c = 0; c < num_contexts; ++c) {
        for (int symbol = 0; symbol < ANS_MAX_ALPHABET_SIZE; ++symbol) {
          builder[c].Add(symbol);
        }
      }
    }

    if (params.initialize_global_state) {
      bool use_prefix_code =
          params.force_huffman || total_tokens < 100 ||
          params.clustering == HistogramParams::ClusteringType::kFastest ||
          ans_fuzzer_friendly_;
      if (!use_prefix_code) {
        bool all_singleton = true;
        for (size_t i = 0; i < num_contexts; i++) {
          if (builder[i].ShannonEntropy() >= 1e-5) {
            all_singleton = false;
          }
        }
        if (all_singleton) {
          use_prefix_code = true;
        }
      }
      codes->use_prefix_code = use_prefix_code;
    }

    if (params.add_fixed_histograms) {
      // TODO(szabadka) Add more fixed histograms.
      // TODO(szabadka) Reduce alphabet size by choosing a non-default
      // uint_config.
      const size_t alphabet_size = ANS_MAX_ALPHABET_SIZE;
      codes->log_alpha_size = 8;
      JXL_ENSURE(alphabet_size == 1u << codes->log_alpha_size);
      static_assert(ANS_MAX_ALPHABET_SIZE <= ANS_TAB_SIZE,
                    "Alphabet does not fit table");
      codes->encoding_info.emplace_back();
      codes->encoding_info.back().resize(alphabet_size);
      codes->encoded_histograms.emplace_back(memory_manager);
      BitWriter* histo_writer = &codes->encoded_histograms.back();
      JXL_RETURN_IF_ERROR(histo_writer->WithMaxBits(
          256 + alphabet_size * 24, LayerType::Header, nullptr,
          [&]() -> Status {
            JXL_ASSIGN_OR_RETURN(
                size_t ans_cost,
                codes->BuildAndStoreANSEncodingData(
                    memory_manager, params.ans_histogram_strategy,
                    Histogram::Flat(alphabet_size, ANS_TAB_SIZE),
                    histo_writer));
            (void)ans_cost;
            return true;
          }));
    }

    // Encode histograms.
    JXL_ASSIGN_OR_RETURN(
        size_t entropy_bits,
        codes->BuildAndStoreEntropyCodes(memory_manager, params, tokens,
                                         builder, writer, layer, aux_out));
    cost += entropy_bits;
    return true;
  };
  if (writer) {
    JXL_RETURN_IF_ERROR(writer->WithMaxBits(
        128 + num_contexts * 40 + max_contexts * 96, layer, aux_out, body,
        /*finished_histogram=*/true));
  } else {
    JXL_RETURN_IF_ERROR(body());
  }

  if (aux_out != nullptr) {
    aux_out->layer(layer).num_clustered_histograms +=
        codes->encoding_info.size();
  }
  return cost;
}

size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes, size_t context_offset,
                   BitWriter* writer) {
  size_t num_extra_bits = 0;
  if (codes.use_prefix_code) {
    for (const auto& token : tokens) {
      uint32_t tok, nbits, bits;
      size_t histo = codes.context_map[context_offset + token.context];
      (token.is_lz77_length ? codes.lz77.length_uint_config
                            : codes.uint_config[histo])
          .Encode(token.value, &tok, &nbits, &bits);
      tok += token.is_lz77_length ? codes.lz77.min_symbol : 0;
      // Combine two calls to the BitWriter. Equivalent to:
      // writer->Write(codes.encoding_info[histo][tok].depth,
      //               codes.encoding_info[histo][tok].bits);
      // writer->Write(nbits, bits);
      uint64_t data = codes.encoding_info[histo][tok].bits;
      data |= static_cast<uint64_t>(bits)
              << codes.encoding_info[histo][tok].depth;
      writer->Write(codes.encoding_info[histo][tok].depth + nbits, data);
      num_extra_bits += nbits;
    }
    return num_extra_bits;
  }
  std::vector<uint64_t> out;
  std::vector<uint8_t> out_nbits;
  out.reserve(tokens.size());
  out_nbits.reserve(tokens.size());
  uint64_t allbits = 0;
  size_t numallbits = 0;
  // Writes in *reversed* order.
  auto addbits = [&](size_t bits, size_t nbits) {
    if (JXL_UNLIKELY(nbits)) {
      JXL_DASSERT(bits >> nbits == 0);
      if (JXL_UNLIKELY(numallbits + nbits > BitWriter::kMaxBitsPerCall)) {
        out.push_back(allbits);
        out_nbits.push_back(numallbits);
        numallbits = allbits = 0;
      }
      allbits <<= nbits;
      allbits |= bits;
      numallbits += nbits;
    }
  };
  const int end = tokens.size();
  ANSCoder ans;
  if (codes.lz77.enabled || codes.context_map.size() > 1) {
    for (int i = end - 1; i >= 0; --i) {
      const Token token = tokens[i];
      const uint8_t histo = codes.context_map[context_offset + token.context];
      uint32_t tok, nbits, bits;
      (token.is_lz77_length ? codes.lz77.length_uint_config
                            : codes.uint_config[histo])
          .Encode(tokens[i].value, &tok, &nbits, &bits);
      tok += token.is_lz77_length ? codes.lz77.min_symbol : 0;
      const ANSEncSymbolInfo& info = codes.encoding_info[histo][tok];
      JXL_DASSERT(info.freq_ > 0);
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      num_extra_bits += nbits;
      uint8_t ans_nbits = 0;
      uint32_t ans_bits = ans.PutSymbol(info, &ans_nbits);
      addbits(ans_bits, ans_nbits);
    }
  } else {
    for (int i = end - 1; i >= 0; --i) {
      uint32_t tok, nbits, bits;
      codes.uint_config[0].Encode(tokens[i].value, &tok, &nbits, &bits);
      const ANSEncSymbolInfo& info = codes.encoding_info[0][tok];
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      num_extra_bits += nbits;
      uint8_t ans_nbits = 0;
      uint32_t ans_bits = ans.PutSymbol(info, &ans_nbits);
      addbits(ans_bits, ans_nbits);
    }
  }
  const uint32_t state = ans.GetState();
  writer->Write(32, state);
  writer->Write(numallbits, allbits);
  for (int i = out.size(); i > 0; --i) {
    writer->Write(out_nbits[i - 1], out[i - 1]);
  }
  return num_extra_bits;
}

Status WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes, size_t context_offset,
                   BitWriter* writer, LayerType layer, AuxOut* aux_out) {
  // Theoretically, we could have 15 prefix code bits + 31 extra bits.
  return writer->WithMaxBits(
      46 * tokens.size() + 32 * 1024 * 4, layer, aux_out, [&] {
        size_t num_extra_bits =
            WriteTokens(tokens, codes, context_offset, writer);
        if (aux_out != nullptr) {
          aux_out->layer(layer).extra_bits += num_extra_bits;
        }
        return true;
      });
}

void SetANSFuzzerFriendly(bool ans_fuzzer_friendly) {
#if JXL_IS_DEBUG_BUILD  // Guard against accidental / malicious changes.
  ans_fuzzer_friendly_ = ans_fuzzer_friendly;
#endif
}

HistogramParams HistogramParams::ForModular(
    const CompressParams& cparams,
    const std::vector<uint8_t>& extra_dc_precision, bool streaming_mode) {
  HistogramParams params;
  params.streaming_mode = streaming_mode;
  if (cparams.speed_tier > SpeedTier::kKitten) {
    params.clustering = HistogramParams::ClusteringType::kFast;
    params.ans_histogram_strategy =
        cparams.speed_tier > SpeedTier::kThunder
            ? HistogramParams::ANSHistogramStrategy::kFast
            : HistogramParams::ANSHistogramStrategy::kApproximate;
    params.lz77_method =
        cparams.modular_mode && cparams.speed_tier <= SpeedTier::kHare
            ? HistogramParams::LZ77Method::kRLE
            : HistogramParams::LZ77Method::kNone;
    // Near-lossless DC, as well as modular mode, require choosing hybrid uint
    // more carefully.
    if ((!extra_dc_precision.empty() && extra_dc_precision[0] != 0) ||
        (cparams.modular_mode && cparams.speed_tier < SpeedTier::kCheetah)) {
      params.uint_method = HistogramParams::HybridUintMethod::kFast;
    } else {
      params.uint_method = HistogramParams::HybridUintMethod::kNone;
    }
  } else if (cparams.speed_tier <= SpeedTier::kTortoise) {
    params.lz77_method = HistogramParams::LZ77Method::kOptimal;
  } else {
    params.lz77_method = HistogramParams::LZ77Method::kLZ77;
  }
  if (cparams.decoding_speed_tier >= 2) {
    params.max_histograms = 12;
  }
    // No predictor requires LZ77 to compress residuals.
    // Effort 3 and lower have forced predictors, so kNone is set.
    if (cparams.options.predictor == Predictor::Zero && cparams.modular_mode) {
        params.lz77_method = cparams.speed_tier >= SpeedTier::kFalcon
            ? HistogramParams::LZ77Method::kNone
            : cparams.speed_tier >= SpeedTier::kHare
            ? HistogramParams::LZ77Method::kRLE
            : cparams.speed_tier >= SpeedTier::kKitten
            ? HistogramParams::LZ77Method::kLZ77
            : HistogramParams::LZ77Method::kOptimal;
    }
  return params;
}
}  // namespace jxl
