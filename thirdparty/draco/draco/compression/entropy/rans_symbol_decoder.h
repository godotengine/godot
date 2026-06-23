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
#ifndef DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_DECODER_H_
#define DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_DECODER_H_

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/entropy/rans_symbol_coding.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/varint_decoding.h"
#include "draco/draco_features.h"

namespace draco {

// A helper class for decoding symbols using the rANS algorithm (see ans.h).
// The class can be used to decode the probability table and the data encoded
// by the RAnsSymbolEncoder. |unique_symbols_bit_length_t| must be the same as
// the one used for the corresponding RAnsSymbolEncoder.
template <int unique_symbols_bit_length_t>
class RAnsSymbolDecoder {
 public:
  RAnsSymbolDecoder() : num_symbols_(0) {}

  // Initialize the decoder and decode the probability table.
  bool Create(DecoderBuffer *buffer);

  uint32_t num_symbols() const { return num_symbols_; }

  // Starts decoding from the buffer. The buffer will be advanced past the
  // encoded data after this call.
  bool StartDecoding(DecoderBuffer *buffer);
  uint32_t DecodeSymbol() { return ans_.rans_read(); }
  void EndDecoding();

 private:
  static constexpr int rans_precision_bits_ =
      ComputeRAnsPrecisionFromUniqueSymbolsBitLength(
          unique_symbols_bit_length_t);
  static constexpr int rans_precision_ = 1 << rans_precision_bits_;

  std::vector<uint32_t> probability_table_;
  uint32_t num_symbols_;
  RAnsDecoder<rans_precision_bits_> ans_;
};

template <int unique_symbols_bit_length_t>
bool RAnsSymbolDecoder<unique_symbols_bit_length_t>::Create(
    DecoderBuffer *buffer) {
  // Check that the DecoderBuffer version is set.
  if (buffer->bitstream_version() == 0) {
    return false;
  }
  // Decode the number of alphabet symbols.
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (buffer->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!buffer->Decode(&num_symbols_)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_symbols_, buffer)) {
      return false;
    }
  }
  // Check that decoded number of symbols is not unreasonably high. Remaining
  // buffer size must be at least |num_symbols| / 64 bytes to contain the
  // probability table. The |prob_data| below is one byte but it can be
  // theoretically stored for each 64th symbol.
  if (num_symbols_ / 64 > buffer->remaining_size()) {
    return false;
  }
  probability_table_.resize(num_symbols_);
  if (num_symbols_ == 0) {
    return true;
  }
  // Decode the table.
  for (uint32_t i = 0; i < num_symbols_; ++i) {
    uint8_t prob_data = 0;
    // Decode the first byte and extract the number of extra bytes we need to
    // get, or the offset to the next symbol with non-zero probability.
    if (!buffer->Decode(&prob_data)) {
      return false;
    }
    // Token is stored in the first two bits of the first byte. Values 0-2 are
    // used to indicate the number of extra bytes, and value 3 is a special
    // symbol used to denote run-length coding of zero probability entries.
    // See rans_symbol_encoder.h for more details.
    const int token = prob_data & 3;
    if (token == 3) {
      const uint32_t offset = prob_data >> 2;
      if (i + offset >= num_symbols_) {
        return false;
      }
      // Set zero probability for all symbols in the specified range.
      for (uint32_t j = 0; j < offset + 1; ++j) {
        probability_table_[i + j] = 0;
      }
      i += offset;
    } else {
      const int extra_bytes = token;
      uint32_t prob = prob_data >> 2;
      for (int b = 0; b < extra_bytes; ++b) {
        uint8_t eb;
        if (!buffer->Decode(&eb)) {
          return false;
        }
        // Shift 8 bits for each extra byte and subtract 2 for the two first
        // bits.
        prob |= static_cast<uint32_t>(eb) << (8 * (b + 1) - 2);
      }
      probability_table_[i] = prob;
    }
  }
  if (!ans_.rans_build_look_up_table(&probability_table_[0], num_symbols_)) {
    return false;
  }
  return true;
}

template <int unique_symbols_bit_length_t>
bool RAnsSymbolDecoder<unique_symbols_bit_length_t>::StartDecoding(
    DecoderBuffer *buffer) {
  uint64_t bytes_encoded;
  // Decode the number of bytes encoded by the encoder.
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (buffer->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!buffer->Decode(&bytes_encoded)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint<uint64_t>(&bytes_encoded, buffer)) {
      return false;
    }
  }
  if (bytes_encoded > static_cast<uint64_t>(buffer->remaining_size())) {
    return false;
  }
  const uint8_t *const data_head =
      reinterpret_cast<const uint8_t *>(buffer->data_head());
  // Advance the buffer past the rANS data.
  buffer->Advance(bytes_encoded);
  if (ans_.read_init(data_head, static_cast<int>(bytes_encoded)) != 0) {
    return false;
  }
  return true;
}

template <int unique_symbols_bit_length_t>
void RAnsSymbolDecoder<unique_symbols_bit_length_t>::EndDecoding() {
  ans_.read_end();
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_DECODER_H_
