// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_UTIL_BITUTILS_H_
#define SOURCE_UTIL_BITUTILS_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace spvtools {
namespace utils {

// Performs a bitwise copy of source to the destination type Dest.
template <typename Dest, typename Src>
Dest BitwiseCast(Src source) {
  Dest dest;
  static_assert(sizeof(source) == sizeof(dest),
                "BitwiseCast: Source and destination must have the same size");
  std::memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// Calculates the bit width of the integer type |T|.
template <typename T>
struct IntegerBitWidth {
  static_assert(std::is_integral<T>::value, "Integer type required");
  static const size_t kBitsPerByte = 8;
  static const size_t get = sizeof(T) * kBitsPerByte;
};

// SetBits<T, First, Num> returns an integer of type <T> with bits set
// for position <First> through <First + Num - 1>, counting from the least
// significant bit. In particular when Num == 0, no positions are set to 1.
// A static assert will be triggered if First + Num > sizeof(T) * 8, that is,
// a bit that will not fit in the underlying type is set.
template <typename T, size_t First = 0, size_t Num = 0>
struct SetBits {
  static_assert(First < IntegerBitWidth<T>::get,
                "Tried to set a bit that is shifted too far.");
  const static T get = (T(1) << First) | SetBits<T, First + 1, Num - 1>::get;
};

template <typename T, size_t Last>
struct SetBits<T, Last, 0> {
  const static T get = T(0);
};

// This is all compile-time so we can put our tests right here.
static_assert(IntegerBitWidth<uint32_t>::get == 32, "IntegerBitWidth mismatch");
static_assert(IntegerBitWidth<int32_t>::get == 32, "IntegerBitWidth mismatch");
static_assert(IntegerBitWidth<uint64_t>::get == 64, "IntegerBitWidth mismatch");
static_assert(IntegerBitWidth<uint8_t>::get == 8, "IntegerBitWidth mismatch");

static_assert(SetBits<uint32_t, 0, 0>::get == uint32_t(0x00000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 0, 1>::get == uint32_t(0x00000001),
              "SetBits failed");
static_assert(SetBits<uint32_t, 31, 1>::get == uint32_t(0x80000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 1, 2>::get == uint32_t(0x00000006),
              "SetBits failed");
static_assert(SetBits<uint32_t, 30, 2>::get == uint32_t(0xc0000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 0, 31>::get == uint32_t(0x7FFFFFFF),
              "SetBits failed");
static_assert(SetBits<uint32_t, 0, 32>::get == uint32_t(0xFFFFFFFF),
              "SetBits failed");
static_assert(SetBits<uint32_t, 16, 16>::get == uint32_t(0xFFFF0000),
              "SetBits failed");

static_assert(SetBits<uint64_t, 0, 1>::get == uint64_t(0x0000000000000001LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 63, 1>::get == uint64_t(0x8000000000000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 62, 2>::get == uint64_t(0xc000000000000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 31, 1>::get == uint64_t(0x0000000080000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 16, 16>::get == uint64_t(0x00000000FFFF0000LL),
              "SetBits failed");

// Returns number of '1' bits in a word.
template <typename T>
size_t CountSetBits(T word) {
  static_assert(std::is_integral<T>::value,
                "CountSetBits requires integer type");
  size_t count = 0;
  while (word) {
    word &= word - 1;
    ++count;
  }
  return count;
}

// Checks if the bit at the |position| is set to '1'.
// Bits zero-indexed starting at the least significant bit.
// |position| must be within the bit width of |T|.
template <typename T>
bool IsBitAtPositionSet(T word, size_t position) {
  static_assert(std::is_integral<T>::value, "Integer type required");
  static_assert(std::is_unsigned<T>::value, "Unsigned type required");
  assert(position < IntegerBitWidth<T>::get &&
         "position must be less than the bit width");
  return word & T(T(1) << position);
}

// Returns a value obtained by setting a range of adjacent bits of |word| to
// |value|. Affected bits are within the range:
//   [first_position, first_position + num_bits_to_mutate),
// assuming zero-based indexing starting at the least
// significant bit. Bits to mutate must be within the bit width of |T|.
template <typename T>
T MutateBits(T word, size_t first_position, size_t num_bits_to_mutate,
             bool value) {
  static_assert(std::is_integral<T>::value, "Integer type required");
  static_assert(std::is_unsigned<T>::value, "Unsigned type required");
  static const size_t word_bit_width = IntegerBitWidth<T>::get;
  assert(first_position < word_bit_width &&
         "Mutated bits must be within bit width");
  assert(first_position + num_bits_to_mutate <= word_bit_width &&
         "Mutated bits must be within bit width");
  if (num_bits_to_mutate == 0) {
    return word;
  }

  const T all_ones = ~T(0);
  const size_t num_unaffected_low_bits = first_position;
  const T unaffected_low_mask =
      T(T(all_ones >> num_unaffected_low_bits) << num_unaffected_low_bits);

  const size_t num_unaffected_high_bits =
      word_bit_width - (first_position + num_bits_to_mutate);
  const T unaffected_high_mask =
      T(T(all_ones << num_unaffected_high_bits) >> num_unaffected_high_bits);

  const T mutation_mask = unaffected_low_mask & unaffected_high_mask;
  if (value) {
    return word | mutation_mask;
  }
  return word & T(~mutation_mask);
}

// Returns a value obtained by setting the |num_bits_to_set| highest bits to
// '1'. |num_bits_to_set| must be not be greater than the bit width of |T|.
template <typename T>
T SetHighBits(T word, size_t num_bits_to_set) {
  if (num_bits_to_set == 0) {
    return word;
  }
  const size_t word_bit_width = IntegerBitWidth<T>::get;
  assert(num_bits_to_set <= word_bit_width &&
         "Can't set more bits than bit width");
  return MutateBits(word, word_bit_width - num_bits_to_set, num_bits_to_set,
                    true);
}

// Returns a value obtained by setting the |num_bits_to_set| highest bits to
// '0'. |num_bits_to_set| must be not be greater than the bit width of |T|.
template <typename T>
T ClearHighBits(T word, size_t num_bits_to_set) {
  if (num_bits_to_set == 0) {
    return word;
  }
  const size_t word_bit_width = IntegerBitWidth<T>::get;
  assert(num_bits_to_set <= word_bit_width &&
         "Can't clear more bits than bit width");
  return MutateBits(word, word_bit_width - num_bits_to_set, num_bits_to_set,
                    false);
}

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_BITUTILS_H_
