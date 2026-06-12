// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_ENCODER_H_
#define DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_ENCODER_H_

#include <algorithm>
#include <cmath>
#include <cstring>

#include "draco/compression/entropy/ans.h"
#include "draco/compression/entropy/rans_symbol_coding.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/varint_encoding.h"

namespace draco {

// A helper class for encoding symbols using the rANS algorithm (see ans.h).
// The class can be used to initialize and encode probability table needed by
// rANS, and to perform encoding of symbols into the provided EncoderBuffer.
template <int unique_symbols_bit_length_t>
class RAnsSymbolEncoder {
 public:
  RAnsSymbolEncoder()
      : num_symbols_(0), num_expected_bits_(0), buffer_offset_(0) {}

  // Creates a probability table needed by the rANS library and encode it into
  // the provided buffer.
  bool Create(const uint64_t *frequencies, int num_symbols,
              EncoderBuffer *buffer);

  void StartEncoding(EncoderBuffer *buffer);
  void EncodeSymbol(uint32_t symbol) {
    ans_.rans_write(&probability_table_[symbol]);
  }
  void EndEncoding(EncoderBuffer *buffer);

  // rANS requires to encode the input symbols in the reverse order.
  static constexpr bool needs_reverse_encoding() { return true; }

 private:
  // Functor used for sorting symbol ids according to their probabilities.
  // The functor sorts symbol indices that index an underlying map between
  // symbol ids and their probabilities. We don't sort the probability table
  // directly, because that would require an additional indirection during the
  // EncodeSymbol() function.
  struct ProbabilityLess {
    explicit ProbabilityLess(const std::vector<rans_sym> *probs)
        : probabilities(probs) {}
    bool operator()(int i, int j) const {
      return probabilities->at(i).prob < probabilities->at(j).prob;
    }
    const std::vector<rans_sym> *probabilities;
  };

  // Encodes the probability table into the output buffer.
  bool EncodeTable(EncoderBuffer *buffer);

  static constexpr int rans_precision_bits_ =
      ComputeRAnsPrecisionFromUniqueSymbolsBitLength(
          unique_symbols_bit_length_t);
  static constexpr int rans_precision_ = 1 << rans_precision_bits_;

  std::vector<rans_sym> probability_table_;
  // The number of symbols in the input alphabet.
  uint32_t num_symbols_;
  // Expected number of bits that is needed to encode the input.
  uint64_t num_expected_bits_;

  RAnsEncoder<rans_precision_bits_> ans_;
  // Initial offset of the encoder buffer before any ans data was encoded.
  uint64_t buffer_offset_;
};

template <int unique_symbols_bit_length_t>
bool RAnsSymbolEncoder<unique_symbols_bit_length_t>::Create(
    const uint64_t *frequencies, int num_symbols, EncoderBuffer *buffer) {
  // Compute the total of the input frequencies.
  uint64_t total_freq = 0;
  int max_valid_symbol = 0;
  for (int i = 0; i < num_symbols; ++i) {
    total_freq += frequencies[i];
    if (frequencies[i] > 0) {
      max_valid_symbol = i;
    }
  }
  num_symbols = max_valid_symbol + 1;
  num_symbols_ = num_symbols;
  probability_table_.resize(num_symbols);
  const double total_freq_d = static_cast<double>(total_freq);
  const double rans_precision_d = static_cast<double>(rans_precision_);
  // Compute probabilities by rescaling the normalized frequencies into interval
  // [1, rans_precision - 1]. The total probability needs to be equal to
  // rans_precision.
  int total_rans_prob = 0;
  for (int i = 0; i < num_symbols; ++i) {
    const uint64_t freq = frequencies[i];

    // Normalized probability.
    const double prob = static_cast<double>(freq) / total_freq_d;

    // RAns probability in range of [1, rans_precision - 1].
    uint32_t rans_prob = static_cast<uint32_t>(prob * rans_precision_d + 0.5f);
    if (rans_prob == 0 && freq > 0) {
      rans_prob = 1;
    }
    probability_table_[i].prob = rans_prob;
    total_rans_prob += rans_prob;
  }
  // Because of rounding errors, the total precision may not be exactly accurate
  // and we may need to adjust the entries a little bit.
  if (total_rans_prob != rans_precision_) {
    std::vector<int> sorted_probabilities(num_symbols);
    for (int i = 0; i < num_symbols; ++i) {
      sorted_probabilities[i] = i;
    }
    std::stable_sort(sorted_probabilities.begin(), sorted_probabilities.end(),
                     ProbabilityLess(&probability_table_));
    if (total_rans_prob < rans_precision_) {
      // This happens rather infrequently, just add the extra needed precision
      // to the most frequent symbol.
      probability_table_[sorted_probabilities.back()].prob +=
          rans_precision_ - total_rans_prob;
    } else {
      // We have over-allocated the precision, which is quite common.
      // Rescale the probabilities of all symbols.
      int32_t error = total_rans_prob - rans_precision_;
      while (error > 0) {
        const double act_total_prob_d = static_cast<double>(total_rans_prob);
        const double act_rel_error_d = rans_precision_d / act_total_prob_d;
        for (int j = num_symbols - 1; j > 0; --j) {
          int symbol_id = sorted_probabilities[j];
          if (probability_table_[symbol_id].prob <= 1) {
            if (j == num_symbols - 1) {
              return false;  // Most frequent symbol would be empty.
            }
            break;
          }
          const int32_t new_prob = static_cast<int32_t>(
              floor(act_rel_error_d *
                    static_cast<double>(probability_table_[symbol_id].prob)));
          int32_t fix = probability_table_[symbol_id].prob - new_prob;
          if (fix == 0u) {
            fix = 1;
          }
          if (fix >= static_cast<int32_t>(probability_table_[symbol_id].prob)) {
            fix = probability_table_[symbol_id].prob - 1;
          }
          if (fix > error) {
            fix = error;
          }
          probability_table_[symbol_id].prob -= fix;
          total_rans_prob -= fix;
          error -= fix;
          if (total_rans_prob == rans_precision_) {
            break;
          }
        }
      }
    }
  }

  // Compute the cumulative probability (cdf).
  uint32_t total_prob = 0;
  for (int i = 0; i < num_symbols; ++i) {
    probability_table_[i].cum_prob = total_prob;
    total_prob += probability_table_[i].prob;
  }
  if (total_prob != rans_precision_) {
    return false;
  }

  // Estimate the number of bits needed to encode the input.
  // From Shannon entropy the total number of bits N is:
  //   N = -sum{i : all_symbols}(F(i) * log2(P(i)))
  // where P(i) is the normalized probability of symbol i and F(i) is the
  // symbol's frequency in the input data.
  double num_bits = 0;
  for (int i = 0; i < num_symbols; ++i) {
    if (probability_table_[i].prob == 0) {
      continue;
    }
    const double norm_prob =
        static_cast<double>(probability_table_[i].prob) / rans_precision_d;
    num_bits += static_cast<double>(frequencies[i]) * log2(norm_prob);
  }
  num_expected_bits_ = static_cast<uint64_t>(ceil(-num_bits));
  if (!EncodeTable(buffer)) {
    return false;
  }
  return true;
}

template <int unique_symbols_bit_length_t>
bool RAnsSymbolEncoder<unique_symbols_bit_length_t>::EncodeTable(
    EncoderBuffer *buffer) {
  EncodeVarint(num_symbols_, buffer);
  // Use varint encoding for the probabilities (first two bits represent the
  // number of bytes used - 1).
  for (uint32_t i = 0; i < num_symbols_; ++i) {
    const uint32_t prob = probability_table_[i].prob;
    int num_extra_bytes = 0;
    if (prob >= (1 << 6)) {
      num_extra_bytes++;
      if (prob >= (1 << 14)) {
        num_extra_bytes++;
        if (prob >= (1 << 22)) {
          // The maximum number of precision bits is 20 so we should not really
          // get to this point.
          return false;
        }
      }
    }
    if (prob == 0) {
      // When the probability of the symbol is 0, set the first two bits to 1
      // (unique identifier) and use the remaining 6 bits to store the offset
      // to the next symbol with non-zero probability.
      uint32_t offset = 0;
      for (; offset < (1 << 6) - 1; ++offset) {
        // Note: we don't have to check whether the next symbol id is larger
        // than num_symbols_ because we know that the last symbol always has
        // non-zero probability.
        const uint32_t next_prob = probability_table_[i + offset + 1].prob;
        if (next_prob > 0) {
          break;
        }
      }
      buffer->Encode(static_cast<uint8_t>((offset << 2) | 3));
      i += offset;
    } else {
      // Encode the first byte (including the number of extra bytes).
      buffer->Encode(static_cast<uint8_t>((prob << 2) | (num_extra_bytes & 3)));
      // Encode the extra bytes.
      for (int b = 0; b < num_extra_bytes; ++b) {
        buffer->Encode(static_cast<uint8_t>(prob >> (8 * (b + 1) - 2)));
      }
    }
  }
  return true;
}

template <int unique_symbols_bit_length_t>
void RAnsSymbolEncoder<unique_symbols_bit_length_t>::StartEncoding(
    EncoderBuffer *buffer) {
  // Allocate extra storage just in case.
  const uint64_t required_bits = 2 * num_expected_bits_ + 32;

  buffer_offset_ = buffer->size();
  const int64_t required_bytes = (required_bits + 7) / 8;
  buffer->Resize(buffer_offset_ + required_bytes + sizeof(buffer_offset_));
  uint8_t *const data =
      reinterpret_cast<uint8_t *>(const_cast<char *>(buffer->data()));
  ans_.write_init(data + buffer_offset_);
}

template <int unique_symbols_bit_length_t>
void RAnsSymbolEncoder<unique_symbols_bit_length_t>::EndEncoding(
    EncoderBuffer *buffer) {
  char *const src = const_cast<char *>(buffer->data()) + buffer_offset_;

  // TODO(fgalligan): Look into changing this to uint32_t as write_end()
  // returns an int.
  const uint64_t bytes_written = static_cast<uint64_t>(ans_.write_end());
  EncoderBuffer var_size_buffer;
  EncodeVarint(bytes_written, &var_size_buffer);
  const uint32_t size_len = static_cast<uint32_t>(var_size_buffer.size());
  char *const dst = src + size_len;
  memmove(dst, src, bytes_written);

  // Store the size of the encoded data.
  memcpy(src, var_size_buffer.data(), size_len);

  // Resize the buffer to match the number of encoded bytes.
  buffer->Resize(buffer_offset_ + bytes_written + size_len);
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_ENCODER_H_
