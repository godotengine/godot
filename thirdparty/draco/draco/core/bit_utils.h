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
// File containing a basic set of bit manipulation utilities used within the
// Draco library.

#ifndef DRACO_CORE_BIT_UTILS_H_
#define DRACO_CORE_BIT_UTILS_H_

#include <inttypes.h>
#include <stdint.h>

#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif  // defined(_MSC_VER)

namespace draco {

// Returns the number of '1' bits within the input 32 bit integer.
inline int CountOneBits32(uint32_t n) {
  n -= ((n >> 1) & 0x55555555);
  n = ((n >> 2) & 0x33333333) + (n & 0x33333333);
  return (((n + (n >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

inline uint32_t ReverseBits32(uint32_t n) {
  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8);
  return (n >> 16) | (n << 16);
}

// Copies the |nbits| from the src integer into the |dst| integer using the
// provided bit offsets |dst_offset| and |src_offset|.
inline void CopyBits32(uint32_t *dst, int dst_offset, uint32_t src,
                       int src_offset, int nbits) {
  const uint32_t mask = (~static_cast<uint32_t>(0)) >> (32 - nbits)
                                                           << dst_offset;
  *dst = (*dst & (~mask)) | (((src >> src_offset) << dst_offset) & mask);
}

// Returns the location of the most significant bit in the input integer |n|.
// The functionality is not defined for |n == 0|.
inline int MostSignificantBit(uint32_t n) {
#if defined(__GNUC__)
  return 31 ^ __builtin_clz(n);
#elif defined(_MSC_VER)

  unsigned long where;
  _BitScanReverse(&where, n);
  return (int)where;
#else
  // TODO(fgalligan): Optimize this code.
  int msb = -1;
  while (n != 0) {
    msb++;
    n >>= 1;
  }
  return msb;
#endif
}

// Helper function that converts signed integer values into unsigned integer
// symbols that can be encoded using an entropy encoder.
void ConvertSignedIntsToSymbols(const int32_t *in, int in_values,
                                uint32_t *out);

// Converts unsigned integer symbols encoded with an entropy encoder back to
// signed values.
void ConvertSymbolsToSignedInts(const uint32_t *in, int in_values,
                                int32_t *out);

// Helper function that converts a single signed integer value into an unsigned
// integer symbol that can be encoded using an entropy encoder.
template <class IntTypeT>
typename std::make_unsigned<IntTypeT>::type ConvertSignedIntToSymbol(
    IntTypeT val) {
  typedef typename std::make_unsigned<IntTypeT>::type UnsignedType;
  static_assert(std::is_integral<IntTypeT>::value, "IntTypeT is not integral.");
  // Early exit if val is positive.
  if (val >= 0) {
    return static_cast<UnsignedType>(val) << 1;
  }
  val = -(val + 1);  // Map -1 to 0, -2 to -1, etc..
  UnsignedType ret = static_cast<UnsignedType>(val);
  ret <<= 1;
  ret |= 1;
  return ret;
}

// Converts a single unsigned integer symbol encoded with an entropy encoder
// back to a signed value.
template <class IntTypeT>
typename std::make_signed<IntTypeT>::type ConvertSymbolToSignedInt(
    IntTypeT val) {
  static_assert(std::is_integral<IntTypeT>::value, "IntTypeT is not integral.");
  typedef typename std::make_signed<IntTypeT>::type SignedType;
  const bool is_positive = !static_cast<bool>(val & 1);
  val >>= 1;
  if (is_positive) {
    return static_cast<SignedType>(val);
  }
  SignedType ret = static_cast<SignedType>(val);
  ret = -ret - 1;
  return ret;
}

}  // namespace draco

#endif  // DRACO_CORE_BIT_UTILS_H_
