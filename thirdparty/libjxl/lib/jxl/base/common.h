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
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#if JXL_COMPILER_MSVC
#include <intrin.h>
#endif

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

template <typename U,
          class = typename std::enable_if<std::is_unsigned<U>::value>::type>
static inline bool SafeAdd(const U a, const U b, U& sum) {
  sum = a + b;
  return sum >= a;  // no need to check b - either sum >= both or < both.
}

static inline bool SafeMul(size_t a, size_t b, size_t& product) {
  product = 0;
  if (a == 0 || b == 0) return true;
  if (b > (std::numeric_limits<size_t>::max() / a)) return false;
  product = a * b;
  return true;
}

static inline bool SubOverflow(const int32_t a, const int32_t b, int32_t& c) {
  // Clang 3.8+ / GCC 5.1+
#if JXL_COMPILER_GCC || JXL_COMPILER_CLANG
  return __builtin_sub_overflow(a, b, &c);
#elif JXL_COMPILER_MSVC >= 1937 && (defined(_M_AMD64) || defined(_M_IX86))
  return _sub_overflow_i32(/*carry*/ 0, a, b, &c);
#else
  uint32_t ua = static_cast<uint32_t>(a);
  uint32_t ub = static_cast<uint32_t>(b);
  uint32_t uc = ua - ub;
  c = static_cast<int32_t>(uc);
  return !!(((ua ^ ub) & (ua ^ uc)) >> 31);
#endif
}

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

// Works for any `align`; if a power of two, compiler emits ADD+AND.
constexpr inline size_t RoundUpTo(size_t what, size_t align) {
  return DivCeil(what, align) * align;
}

// `align <= 1` means no rounding.
static inline bool SafeRoundUpTo(size_t what, size_t align, size_t& result) {
  if (align < 2) {
    result = what;
    return true;
  }
  size_t reminder = what % align;
  if (reminder == 0) {
    result = what;
    return true;
  }
  return SafeAdd(what, align - reminder, result);
}

constexpr double kPi = 3.14159265358979323846264338327950288;

// Multiplier for conversion of log2(x) result to ln(x).
// print(1.0 / math.log2(math.e))
constexpr float kInvLog2e = 0.6931471805599453;

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

template <typename T>
struct UninitializedAllocator : std::allocator<T> {
  static_assert(std::is_trivially_copyable<T>::value,
                "Uninitialized values have to be trivially destructible");
  using value_type = T;

  UninitializedAllocator() noexcept = default;
  UninitializedAllocator(const UninitializedAllocator& other) noexcept =
      default;

  template <typename U>
  explicit UninitializedAllocator(
      const UninitializedAllocator<U>& other) noexcept {}

  template <typename U>
  struct rebind {
    using other = UninitializedAllocator<U>;
  };

  template <typename U, typename... Args>
  void construct(U* place, Args&&... args) {}

  template <typename U>
  void destroy(U* place) {}
};

template <typename T>
using uninitialized_vector = std::vector<T, UninitializedAllocator<T>>;

template <typename T>
uninitialized_vector<T> make_uninitialized_vector(size_t n) {
  return uninitialized_vector<T>(n, UninitializedAllocator<T>());
}

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
