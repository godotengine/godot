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
// File providing shared functionality for RAnsSymbolEncoder and
// RAnsSymbolDecoder (see rans_symbol_encoder.h / rans_symbol_decoder.h).
#ifndef DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_CODING_H_
#define DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_CODING_H_

#include "draco/compression/entropy/ans.h"

namespace draco {

// Computes the desired precision of the rANS method for the specified number of
// unique symbols the input data (defined by their bit_length).
constexpr int ComputeRAnsUnclampedPrecision(int symbols_bit_length) {
  return (3 * symbols_bit_length) / 2;
}

// Computes the desired precision clamped to guarantee a valid functionality of
// our rANS library (which is between 12 to 20 bits).
constexpr int ComputeRAnsPrecisionFromUniqueSymbolsBitLength(
    int symbols_bit_length) {
  return ComputeRAnsUnclampedPrecision(symbols_bit_length) < 12 ? 12
         : ComputeRAnsUnclampedPrecision(symbols_bit_length) > 20
             ? 20
             : ComputeRAnsUnclampedPrecision(symbols_bit_length);
}

// Compute approximate frequency table size needed for storing the provided
// symbols.
static inline int64_t ApproximateRAnsFrequencyTableBits(
    int32_t max_value, int num_unique_symbols) {
  // Approximate number of bits for storing zero frequency entries using the
  // run length encoding (with max length of 64).
  const int64_t table_zero_frequency_bits =
      8 * (num_unique_symbols + (max_value - num_unique_symbols) / 64);
  return 8 * num_unique_symbols + table_zero_frequency_bits;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_RANS_SYMBOL_CODING_H_
