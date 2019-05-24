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

#include <cstdint>
#include <cstring>

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

// SetBits<T, First, Num> returns an integer of type <T> with bits set
// for position <First> through <First + Num - 1>, counting from the least
// significant bit. In particular when Num == 0, no positions are set to 1.
// A static assert will be triggered if First + Num > sizeof(T) * 8, that is,
// a bit that will not fit in the underlying type is set.
template <typename T, size_t First = 0, size_t Num = 0>
struct SetBits {
  static_assert(First < sizeof(T) * 8,
                "Tried to set a bit that is shifted too far.");
  const static T get = (T(1) << First) | SetBits<T, First + 1, Num - 1>::get;
};

template <typename T, size_t Last>
struct SetBits<T, Last, 0> {
  const static T get = T(0);
};

// This is all compile-time so we can put our tests right here.
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

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_BITUTILS_H_
