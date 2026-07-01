// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_BITS_H_
#define LIB_JXL_BASE_BITS_H_

// Specialized instructions for processing register-sized bit arrays.

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

#if JXL_COMPILER_MSVC
#include <intrin.h>
#endif

#include <stddef.h>
#include <stdint.h>

namespace jxl {

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

template <typename T>
constexpr bool IsSigned() {
  // TODO(eustas): remove dupes
  return static_cast<T>(0) > static_cast<T>(-1);
}

// Undefined results for x == 0.
static JXL_INLINE JXL_MAYBE_UNUSED size_t
Num0BitsAboveMS1Bit_Nonzero(SizeTag<4> /* tag */, const uint32_t x) {
  JXL_DASSERT(x != 0);
#if JXL_COMPILER_MSVC
  unsigned long index;
  _BitScanReverse(&index, x);
  return 31 - index;
#else
  return static_cast<size_t>(__builtin_clz(x));
#endif
}
static JXL_INLINE JXL_MAYBE_UNUSED size_t
Num0BitsAboveMS1Bit_Nonzero(SizeTag<8> /* tag */, const uint64_t x) {
  JXL_DASSERT(x != 0);
#if JXL_COMPILER_MSVC
#if JXL_ARCH_X64
  unsigned long index;
  _BitScanReverse64(&index, x);
  return 63 - index;
#else   // JXL_ARCH_X64
  // _BitScanReverse64 not available
  uint32_t msb = static_cast<uint32_t>(x >> 32u);
  unsigned long index;
  if (msb == 0) {
    uint32_t lsb = static_cast<uint32_t>(x & 0xFFFFFFFF);
    _BitScanReverse(&index, lsb);
    return 63 - index;
  } else {
    _BitScanReverse(&index, msb);
    return 31 - index;
  }
#endif  // JXL_ARCH_X64
#else
  return static_cast<size_t>(__builtin_clzll(x));
#endif
}
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t
Num0BitsAboveMS1Bit_Nonzero(const T x) {
  static_assert(!IsSigned<T>(), "Num0BitsAboveMS1Bit_Nonzero: use unsigned");
  return Num0BitsAboveMS1Bit_Nonzero(SizeTag<sizeof(T)>(), x);
}

// Undefined results for x == 0.
static JXL_INLINE JXL_MAYBE_UNUSED size_t
Num0BitsBelowLS1Bit_Nonzero(SizeTag<4> /* tag */, const uint32_t x) {
  JXL_DASSERT(x != 0);
#if JXL_COMPILER_MSVC
  unsigned long index;
  _BitScanForward(&index, x);
  return index;
#else
  return static_cast<size_t>(__builtin_ctz(x));
#endif
}
static JXL_INLINE JXL_MAYBE_UNUSED size_t
Num0BitsBelowLS1Bit_Nonzero(SizeTag<8> /* tag */, const uint64_t x) {
  JXL_DASSERT(x != 0);
#if JXL_COMPILER_MSVC
#if JXL_ARCH_X64
  unsigned long index;
  _BitScanForward64(&index, x);
  return index;
#else   // JXL_ARCH_64
  // _BitScanForward64 not available
  uint32_t lsb = static_cast<uint32_t>(x & 0xFFFFFFFF);
  unsigned long index;
  if (lsb == 0) {
    uint32_t msb = static_cast<uint32_t>(x >> 32u);
    _BitScanForward(&index, msb);
    return 32 + index;
  } else {
    _BitScanForward(&index, lsb);
    return index;
  }
#endif  // JXL_ARCH_X64
#else
  return static_cast<size_t>(__builtin_ctzll(x));
#endif
}
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t Num0BitsBelowLS1Bit_Nonzero(T x) {
  static_assert(!IsSigned<T>(), "Num0BitsBelowLS1Bit_Nonzero: use unsigned");
  return Num0BitsBelowLS1Bit_Nonzero(SizeTag<sizeof(T)>(), x);
}

// Returns bit width for x == 0.
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t Num0BitsAboveMS1Bit(const T x) {
  return (x == 0) ? sizeof(T) * 8 : Num0BitsAboveMS1Bit_Nonzero(x);
}

// Returns bit width for x == 0.
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t Num0BitsBelowLS1Bit(const T x) {
  return (x == 0) ? sizeof(T) * 8 : Num0BitsBelowLS1Bit_Nonzero(x);
}

// Returns base-2 logarithm, rounded down.
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t FloorLog2Nonzero(const T x) {
  return (sizeof(T) * 8 - 1) ^ Num0BitsAboveMS1Bit_Nonzero(x);
}

// Returns base-2 logarithm, rounded up.
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED size_t CeilLog2Nonzero(const T x) {
  const size_t floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_BITS_H_
