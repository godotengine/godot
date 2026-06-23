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
#include "draco/compression/entropy/symbol_encoding.h"

#include <algorithm>
#include <cmath>

#include "draco/compression/entropy/rans_symbol_encoder.h"
#include "draco/compression/entropy/shannon_entropy.h"
#include "draco/core/bit_utils.h"
#include "draco/core/macros.h"

namespace draco {

constexpr int32_t kMaxTagSymbolBitLength = 32;
constexpr int kMaxRawEncodingBitLength = 18;
constexpr int kDefaultSymbolCodingCompressionLevel = 7;

typedef uint64_t TaggedBitLengthFrequencies[kMaxTagSymbolBitLength];

void SetSymbolEncodingMethod(Options *options, SymbolCodingMethod method) {
  options->SetInt("symbol_encoding_method", method);
}

bool SetSymbolEncodingCompressionLevel(Options *options,
                                       int compression_level) {
  if (compression_level < 0 || compression_level > 10) {
    return false;
  }
  options->SetInt("symbol_encoding_compression_level", compression_level);
  return true;
}

// Computes bit lengths of the input values. If num_components > 1, the values
// are processed in "num_components" sized chunks and the bit length is always
// computed for the largest value from the chunk.
static void ComputeBitLengths(const uint32_t *symbols, int num_values,
                              int num_components,
                              std::vector<uint32_t> *out_bit_lengths,
                              uint32_t *out_max_value) {
  out_bit_lengths->reserve(num_values);
  *out_max_value = 0;
  // Maximum integer value across all components.
  for (int i = 0; i < num_values; i += num_components) {
    // Get the maximum value for a given entry across all attribute components.
    uint32_t max_component_value = symbols[i];
    for (int j = 1; j < num_components; ++j) {
      if (max_component_value < symbols[i + j]) {
        max_component_value = symbols[i + j];
      }
    }
    int value_msb_pos = 0;
    if (max_component_value > 0) {
      value_msb_pos = MostSignificantBit(max_component_value);
    }
    if (max_component_value > *out_max_value) {
      *out_max_value = max_component_value;
    }
    out_bit_lengths->push_back(value_msb_pos + 1);
  }
}

static int64_t ApproximateTaggedSchemeBits(
    const std::vector<uint32_t> bit_lengths, int num_components) {
  // Compute the total bit length used by all values (the length of data encode
  // after tags).
  uint64_t total_bit_length = 0;
  for (size_t i = 0; i < bit_lengths.size(); ++i) {
    total_bit_length += bit_lengths[i];
  }
  // Compute the number of entropy bits for tags.
  int num_unique_symbols;
  const int64_t tag_bits = ComputeShannonEntropy(
      bit_lengths.data(), static_cast<int>(bit_lengths.size()), 32,
      &num_unique_symbols);
  const int64_t tag_table_bits =
      ApproximateRAnsFrequencyTableBits(num_unique_symbols, num_unique_symbols);
  return tag_bits + tag_table_bits + total_bit_length * num_components;
}

static int64_t ApproximateRawSchemeBits(const uint32_t *symbols,
                                        int num_symbols, uint32_t max_value,
                                        int *out_num_unique_symbols) {
  int num_unique_symbols;
  const int64_t data_bits = ComputeShannonEntropy(
      symbols, num_symbols, max_value, &num_unique_symbols);
  const int64_t table_bits =
      ApproximateRAnsFrequencyTableBits(max_value, num_unique_symbols);
  *out_num_unique_symbols = num_unique_symbols;
  return table_bits + data_bits;
}

template <template <int> class SymbolEncoderT>
bool EncodeTaggedSymbols(const uint32_t *symbols, int num_values,
                         int num_components,
                         const std::vector<uint32_t> &bit_lengths,
                         EncoderBuffer *target_buffer);

template <template <int> class SymbolEncoderT>
bool EncodeRawSymbols(const uint32_t *symbols, int num_values,
                      uint32_t max_entry_value, int32_t num_unique_symbols,
                      const Options *options, EncoderBuffer *target_buffer);

bool EncodeSymbols(const uint32_t *symbols, int num_values, int num_components,
                   const Options *options, EncoderBuffer *target_buffer) {
  if (num_values < 0) {
    return false;
  }
  if (num_values == 0) {
    return true;
  }
  if (num_components <= 0) {
    num_components = 1;
  }
  std::vector<uint32_t> bit_lengths;
  uint32_t max_value;
  ComputeBitLengths(symbols, num_values, num_components, &bit_lengths,
                    &max_value);

  // Approximate number of bits needed for storing the symbols using the tagged
  // scheme.
  const int64_t tagged_scheme_total_bits =
      ApproximateTaggedSchemeBits(bit_lengths, num_components);

  // Approximate number of bits needed for storing the symbols using the raw
  // scheme.
  int num_unique_symbols = 0;
  const int64_t raw_scheme_total_bits = ApproximateRawSchemeBits(
      symbols, num_values, max_value, &num_unique_symbols);

  // The maximum bit length of a single entry value that we can encode using
  // the raw scheme.
  const int max_value_bit_length =
      MostSignificantBit(std::max(1u, max_value)) + 1;

  int method = -1;
  if (options != nullptr && options->IsOptionSet("symbol_encoding_method")) {
    method = options->GetInt("symbol_encoding_method");
  } else {
    if (tagged_scheme_total_bits < raw_scheme_total_bits ||
        max_value_bit_length > kMaxRawEncodingBitLength) {
      method = SYMBOL_CODING_TAGGED;
    } else {
      method = SYMBOL_CODING_RAW;
    }
  }
  // Use the tagged scheme.
  target_buffer->Encode(static_cast<uint8_t>(method));
  if (method == SYMBOL_CODING_TAGGED) {
    return EncodeTaggedSymbols<RAnsSymbolEncoder>(
        symbols, num_values, num_components, bit_lengths, target_buffer);
  }
  if (method == SYMBOL_CODING_RAW) {
    return EncodeRawSymbols<RAnsSymbolEncoder>(symbols, num_values, max_value,
                                               num_unique_symbols, options,
                                               target_buffer);
  }
  // Unknown method selected.
  return false;
}

template <template <int> class SymbolEncoderT>
bool EncodeTaggedSymbols(const uint32_t *symbols, int num_values,
                         int num_components,
                         const std::vector<uint32_t> &bit_lengths,
                         EncoderBuffer *target_buffer) {
  // Create entries for entropy coding. Each entry corresponds to a different
  // number of bits that are necessary to encode a given value. Every value
  // has at most 32 bits. Therefore, we need 32 different entries (for
  // bit_length [1-32]). For each entry we compute the frequency of a given
  // bit-length in our data set.
  TaggedBitLengthFrequencies frequencies;
  // Set frequency for each entry to zero.
  memset(frequencies, 0, sizeof(frequencies));

  // Compute the frequencies from input data.
  // Maximum integer value for the values across all components.
  for (size_t i = 0; i < bit_lengths.size(); ++i) {
    // Update the frequency of the associated entry id.
    ++frequencies[bit_lengths[i]];
  }

  // Create one extra buffer to store raw value.
  EncoderBuffer value_buffer;
  // Number of expected bits we need to store the values (can be optimized if
  // needed).
  const uint64_t value_bits =
      kMaxTagSymbolBitLength * static_cast<uint64_t>(num_values);

  // Create encoder for encoding the bit tags.
  SymbolEncoderT<5> tag_encoder;
  tag_encoder.Create(frequencies, kMaxTagSymbolBitLength, target_buffer);

  // Start encoding bit tags.
  tag_encoder.StartEncoding(target_buffer);

  // Also start encoding the values.
  value_buffer.StartBitEncoding(value_bits, false);

  if (tag_encoder.needs_reverse_encoding()) {
    // Encoder needs the values to be encoded in the reverse order.
    for (int i = num_values - num_components; i >= 0; i -= num_components) {
      const int bit_length = bit_lengths[i / num_components];
      tag_encoder.EncodeSymbol(bit_length);

      // Values are always encoded in the normal order
      const int j = num_values - num_components - i;
      const int value_bit_length = bit_lengths[j / num_components];
      for (int c = 0; c < num_components; ++c) {
        value_buffer.EncodeLeastSignificantBits32(value_bit_length,
                                                  symbols[j + c]);
      }
    }
  } else {
    for (int i = 0; i < num_values; i += num_components) {
      const int bit_length = bit_lengths[i / num_components];
      // First encode the tag.
      tag_encoder.EncodeSymbol(bit_length);
      // Now encode all values using the stored bit_length.
      for (int j = 0; j < num_components; ++j) {
        value_buffer.EncodeLeastSignificantBits32(bit_length, symbols[i + j]);
      }
    }
  }
  tag_encoder.EndEncoding(target_buffer);
  value_buffer.EndBitEncoding();

  // Append the values to the end of the target buffer.
  target_buffer->Encode(value_buffer.data(), value_buffer.size());
  return true;
}

template <class SymbolEncoderT>
bool EncodeRawSymbolsInternal(const uint32_t *symbols, int num_values,
                              uint32_t max_entry_value,
                              EncoderBuffer *target_buffer) {
  // Count the frequency of each entry value.
  std::vector<uint64_t> frequencies(max_entry_value + 1, 0);
  for (int i = 0; i < num_values; ++i) {
    ++frequencies[symbols[i]];
  }

  SymbolEncoderT encoder;
  encoder.Create(frequencies.data(), static_cast<int>(frequencies.size()),
                 target_buffer);
  encoder.StartEncoding(target_buffer);
  // Encode all values.
  if (SymbolEncoderT::needs_reverse_encoding()) {
    for (int i = num_values - 1; i >= 0; --i) {
      encoder.EncodeSymbol(symbols[i]);
    }
  } else {
    for (int i = 0; i < num_values; ++i) {
      encoder.EncodeSymbol(symbols[i]);
    }
  }
  encoder.EndEncoding(target_buffer);
  return true;
}

template <template <int> class SymbolEncoderT>
bool EncodeRawSymbols(const uint32_t *symbols, int num_values,
                      uint32_t max_entry_value, int32_t num_unique_symbols,
                      const Options *options, EncoderBuffer *target_buffer) {
  int symbol_bits = 0;
  if (num_unique_symbols > 0) {
    symbol_bits = MostSignificantBit(num_unique_symbols);
  }
  int unique_symbols_bit_length = symbol_bits + 1;
  // Currently, we don't support encoding of more than 2^18 unique symbols.
  if (unique_symbols_bit_length > kMaxRawEncodingBitLength) {
    return false;
  }
  int compression_level = kDefaultSymbolCodingCompressionLevel;
  if (options != nullptr &&
      options->IsOptionSet("symbol_encoding_compression_level")) {
    compression_level = options->GetInt("symbol_encoding_compression_level");
  }

  // Adjust the bit_length based on compression level. Lower compression levels
  // will use fewer bits while higher compression levels use more bits. Note
  // that this is going to work for all valid bit_lengths because the actual
  // number of bits allocated for rANS encoding is hard coded as:
  // std::max(12, 3 * bit_length / 2) , therefore there will be always a
  // sufficient number of bits available for all symbols.
  // See ComputeRAnsPrecisionFromUniqueSymbolsBitLength() for the formula.
  // This hardcoded equation cannot be changed without changing the bitstream.
  if (compression_level < 4) {
    unique_symbols_bit_length -= 2;
  } else if (compression_level < 6) {
    unique_symbols_bit_length -= 1;
  } else if (compression_level > 9) {
    unique_symbols_bit_length += 2;
  } else if (compression_level > 7) {
    unique_symbols_bit_length += 1;
  }
  // Clamp the bit_length to a valid range.
  unique_symbols_bit_length = std::min(std::max(1, unique_symbols_bit_length),
                                       kMaxRawEncodingBitLength);
  target_buffer->Encode(static_cast<uint8_t>(unique_symbols_bit_length));
  // Use appropriate symbol encoder based on the maximum symbol bit length.
  switch (unique_symbols_bit_length) {
    case 0:
      FALLTHROUGH_INTENDED;
    case 1:
      return EncodeRawSymbolsInternal<SymbolEncoderT<1>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 2:
      return EncodeRawSymbolsInternal<SymbolEncoderT<2>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 3:
      return EncodeRawSymbolsInternal<SymbolEncoderT<3>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 4:
      return EncodeRawSymbolsInternal<SymbolEncoderT<4>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 5:
      return EncodeRawSymbolsInternal<SymbolEncoderT<5>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 6:
      return EncodeRawSymbolsInternal<SymbolEncoderT<6>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 7:
      return EncodeRawSymbolsInternal<SymbolEncoderT<7>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 8:
      return EncodeRawSymbolsInternal<SymbolEncoderT<8>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 9:
      return EncodeRawSymbolsInternal<SymbolEncoderT<9>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 10:
      return EncodeRawSymbolsInternal<SymbolEncoderT<10>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 11:
      return EncodeRawSymbolsInternal<SymbolEncoderT<11>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 12:
      return EncodeRawSymbolsInternal<SymbolEncoderT<12>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 13:
      return EncodeRawSymbolsInternal<SymbolEncoderT<13>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 14:
      return EncodeRawSymbolsInternal<SymbolEncoderT<14>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 15:
      return EncodeRawSymbolsInternal<SymbolEncoderT<15>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 16:
      return EncodeRawSymbolsInternal<SymbolEncoderT<16>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 17:
      return EncodeRawSymbolsInternal<SymbolEncoderT<17>>(
          symbols, num_values, max_entry_value, target_buffer);
    case 18:
      return EncodeRawSymbolsInternal<SymbolEncoderT<18>>(
          symbols, num_values, max_entry_value, target_buffer);
    default:
      return false;
  }
}

}  // namespace draco
