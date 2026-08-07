// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_COMMON_H_
#define LIB_JXL_BASE_COMMON_H_

// Shared constants and helper functions.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {
// Some enums and typedefs used by more than one header file.

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

constexpr inline size_t RoundUpBitsToByteMultiple(size_t bits) {
  return (bits + 7) & ~static_cast<size_t>(7);
}

constexpr inline size_t RoundUpToBlockDim(size_t dim) {
  return (dim + 7) & ~static_cast<size_t>(7);
}

static inline bool JXL_MAYBE_UNUSED SafeAdd(const uint64_t a, const uint64_t b,
                                            uint64_t& sum) {
  sum = a + b;
  return sum >= a;  // no need to check b - either sum >= both or < both.
}

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

// Works for any `align`; if a power of two, compiler emits ADD+AND.
constexpr inline size_t RoundUpTo(size_t what, size_t align) {
  return DivCeil(what, align) * align;
}

constexpr double kPi = 3.14159265358979323846264338327950288;

// Reasonable default for sRGB, matches common monitors. We map white to this
// many nits (cd/m^2) by default. Butteraugli was tuned for 250 nits, which is
// very close.
// NB: This constant is not very "base", but it is shared between modules.
static constexpr float kDefaultIntensityTarget = 255;

template <typename T>
constexpr T Pi(T multiplier) {
  return static_cast<T>(multiplier * kPi);
}

// Prior to C++14 (i.e. C++11): provide our own make_unique
#if __cplusplus < 201402L
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#else
using std::make_unique;
#endif

typedef std::array<float, 3> Color;

// Backported std::experimental::to_array

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <size_t... I>
struct index_sequence {};

template <size_t N, size_t... I>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, I...> {};

template <size_t... I>
struct make_index_sequence<0, I...> : index_sequence<I...> {};

namespace detail {

template <typename T, size_t N, size_t... I>
constexpr auto to_array(T (&&arr)[N], index_sequence<I...> _)
    -> std::array<remove_cv_t<T>, N> {
  return {{std::move(arr[I])...}};
}

}  // namespace detail

template <typename T, size_t N>
constexpr auto to_array(T (&&arr)[N]) -> std::array<remove_cv_t<T>, N> {
  return detail::to_array(std::move(arr), make_index_sequence<N>());
}

template <typename T>
JXL_INLINE T Clamp1(T val, T low, T hi) {
  return val < low ? low : val > hi ? hi : val;
}

// conversion from integer to string.
template <typename T>
std::string ToString(T n) {
  char data[32] = {};
  if (std::is_floating_point<T>::value) {
    // float
    snprintf(data, sizeof(data), "%g", static_cast<double>(n));
  } else if (std::is_unsigned<T>::value) {
    // unsigned
    snprintf(data, sizeof(data), "%llu", static_cast<unsigned long long>(n));
  } else {
    // signed
    snprintf(data, sizeof(data), "%lld", static_cast<long long>(n));
  }
  return data;
}

#define JXL_JOIN(x, y) JXL_DO_JOIN(x, y)
#define JXL_DO_JOIN(x, y) x##y

}  // namespace jxl

#endif  // LIB_JXL_BASE_COMMON_H_
