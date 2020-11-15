// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_IMPL_H_
#define MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "base/numerics/safe_conversions.h"

namespace base {
namespace internal {

// Everything from here up to the floating point operations is portable C++,
// but it may not be fast. This code could be split based on
// platform/architecture and replaced with potentially faster implementations.

// This is used for UnsignedAbs, where we need to support floating-point
// template instantiations even though we don't actually support the operations.
// However, there is no corresponding implementation of e.g. SafeUnsignedAbs,
// so the float versions will not compile.
template <typename Numeric,
          bool IsInteger = std::is_integral<Numeric>::value,
          bool IsFloat = std::is_floating_point<Numeric>::value>
struct UnsignedOrFloatForSize;

template <typename Numeric>
struct UnsignedOrFloatForSize<Numeric, true, false> {
  using type = typename std::make_unsigned<Numeric>::type;
};

template <typename Numeric>
struct UnsignedOrFloatForSize<Numeric, false, true> {
  using type = Numeric;
};

// Helper templates for integer manipulations.

template <typename T>
constexpr bool HasSignBit(T x) {
  // Cast to unsigned since right shift on signed is undefined.
  return !!(static_cast<typename std::make_unsigned<T>::type>(x) >>
            PositionOfSignBit<T>::value);
}

// This wrapper undoes the standard integer promotions.
template <typename T>
constexpr T BinaryComplement(T x) {
  return static_cast<T>(~x);
}

// Probe for builtin math overflow support on Clang and version check on GCC.
#if defined(__has_builtin)
#define USE_OVERFLOW_BUILTINS (__has_builtin(__builtin_add_overflow))
#elif defined(__GNUC__)
#define USE_OVERFLOW_BUILTINS (__GNUC__ >= 5)
#else
#define USE_OVERFLOW_BUILTINS (0)
#endif

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool CheckedAddImpl(T x, T y, T* result) {
  // Since the value of x+y is undefined if we have a signed type, we compute
  // it using the unsigned type of the same size.
  using UnsignedDst = typename std::make_unsigned<T>::type;
  UnsignedDst ux = static_cast<UnsignedDst>(x);
  UnsignedDst uy = static_cast<UnsignedDst>(y);
  UnsignedDst uresult = static_cast<UnsignedDst>(ux + uy);
  *result = static_cast<T>(uresult);
  // Addition is valid if the sign of (x + y) is equal to either that of x or
  // that of y.
  return (std::is_signed<T>::value)
             ? HasSignBit(BinaryComplement(
                   static_cast<UnsignedDst>((uresult ^ ux) & (uresult ^ uy))))
             : (BinaryComplement(x) >=
                y);  // Unsigned is either valid or underflow.
}

template <typename T, typename U, class Enable = void>
struct CheckedAddOp {};

template <typename T, typename U>
struct CheckedAddOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V>
  static bool Do(T x, U y, V* result) {
#if USE_OVERFLOW_BUILTINS
    return !__builtin_add_overflow(x, y, result);
#else
    using Promotion = typename BigEnoughPromotion<T, U>::type;
    Promotion presult;
    // Fail if either operand is out of range for the promoted type.
    // TODO(jschuh): This could be made to work for a broader range of values.
    bool is_valid = IsValueInRangeForNumericType<Promotion>(x) &&
                    IsValueInRangeForNumericType<Promotion>(y);

    if (IsIntegerArithmeticSafe<Promotion, T, U>::value) {
      presult = static_cast<Promotion>(x) + static_cast<Promotion>(y);
    } else {
      is_valid &= CheckedAddImpl(static_cast<Promotion>(x),
                                 static_cast<Promotion>(y), &presult);
    }
    *result = static_cast<V>(presult);
    return is_valid && IsValueInRangeForNumericType<V>(presult);
#endif
  }
};

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool CheckedSubImpl(T x, T y, T* result) {
  // Since the value of x+y is undefined if we have a signed type, we compute
  // it using the unsigned type of the same size.
  using UnsignedDst = typename std::make_unsigned<T>::type;
  UnsignedDst ux = static_cast<UnsignedDst>(x);
  UnsignedDst uy = static_cast<UnsignedDst>(y);
  UnsignedDst uresult = static_cast<UnsignedDst>(ux - uy);
  *result = static_cast<T>(uresult);
  // Subtraction is valid if either x and y have same sign, or (x-y) and x have
  // the same sign.
  return (std::is_signed<T>::value)
             ? HasSignBit(BinaryComplement(
                   static_cast<UnsignedDst>((uresult ^ ux) & (ux ^ uy))))
             : (x >= y);
}

template <typename T, typename U, class Enable = void>
struct CheckedSubOp {};

template <typename T, typename U>
struct CheckedSubOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V>
  static bool Do(T x, U y, V* result) {
#if USE_OVERFLOW_BUILTINS
    return !__builtin_sub_overflow(x, y, result);
#else
    using Promotion = typename BigEnoughPromotion<T, U>::type;
    Promotion presult;
    // Fail if either operand is out of range for the promoted type.
    // TODO(jschuh): This could be made to work for a broader range of values.
    bool is_valid = IsValueInRangeForNumericType<Promotion>(x) &&
                    IsValueInRangeForNumericType<Promotion>(y);

    if (IsIntegerArithmeticSafe<Promotion, T, U>::value) {
      presult = static_cast<Promotion>(x) - static_cast<Promotion>(y);
    } else {
      is_valid &= CheckedSubImpl(static_cast<Promotion>(x),
                                 static_cast<Promotion>(y), &presult);
    }
    *result = static_cast<V>(presult);
    return is_valid && IsValueInRangeForNumericType<V>(presult);
#endif
  }
};

// Integer multiplication is a bit complicated. In the fast case we just
// we just promote to a twice wider type, and range check the result. In the
// slow case we need to manually check that the result won't be truncated by
// checking with division against the appropriate bound.
template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value &&
              ((IntegerBitsPlusSign<T>::value * 2) <=
               IntegerBitsPlusSign<intmax_t>::value)>::type* = nullptr>
bool CheckedMulImpl(T x, T y, T* result) {
  using IntermediateType = typename TwiceWiderInteger<T>::type;
  IntermediateType tmp =
      static_cast<IntermediateType>(x) * static_cast<IntermediateType>(y);
  *result = static_cast<T>(tmp);
  return DstRangeRelationToSrcRange<T>(tmp) == RANGE_VALID;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value &&
              ((IntegerBitsPlusSign<T>::value * 2) >
               IntegerBitsPlusSign<intmax_t>::value)>::type* = nullptr>
bool CheckedMulImpl(T x, T y, T* result) {
  if (x && y) {
    if (x > 0) {
      if (y > 0) {
        if (x > std::numeric_limits<T>::max() / y)
          return false;
      } else {
        if (y < std::numeric_limits<T>::lowest() / x)
          return false;
      }
    } else {
      if (y > 0) {
        if (x < std::numeric_limits<T>::lowest() / y)
          return false;
      } else {
        if (y < std::numeric_limits<T>::max() / x)
          return false;
      }
    }
  }
  *result = x * y;
  return true;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && !std::is_signed<T>::value &&
              ((IntegerBitsPlusSign<T>::value * 2) >
               IntegerBitsPlusSign<uintmax_t>::value)>::type* = nullptr>
bool CheckedMulImpl(T x, T y, T* result) {
  *result = x * y;
  return (y == 0 || x <= std::numeric_limits<T>::max() / y);
}

template <typename T, typename U, class Enable = void>
struct CheckedMulOp {};

template <typename T, typename U>
struct CheckedMulOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V>
  static bool Do(T x, U y, V* result) {
#if USE_OVERFLOW_BUILTINS
#if defined(__clang__)
    // TODO(jschuh): Get the Clang runtime library issues sorted out so we can
    // support full-width, mixed-sign multiply builtins.
    // https://crbug.com/613003
    static const bool kUseMaxInt =
        // Narrower type than uintptr_t is always safe.
        std::numeric_limits<__typeof__(x * y)>::digits <
            std::numeric_limits<intptr_t>::digits ||
        // Safe for intptr_t and uintptr_t if the sign matches.
        (IntegerBitsPlusSign<__typeof__(x * y)>::value ==
             IntegerBitsPlusSign<intptr_t>::value &&
         std::is_signed<T>::value == std::is_signed<U>::value);
#else
    static const bool kUseMaxInt = true;
#endif
    if (kUseMaxInt)
      return !__builtin_mul_overflow(x, y, result);
#endif
    using Promotion = typename BigEnoughPromotion<T, U>::type;
    Promotion presult;
    // Fail if either operand is out of range for the promoted type.
    // TODO(jschuh): This could be made to work for a broader range of values.
    bool is_valid = IsValueInRangeForNumericType<Promotion>(x) &&
                    IsValueInRangeForNumericType<Promotion>(y);

    if (IsIntegerArithmeticSafe<Promotion, T, U>::value) {
      presult = static_cast<Promotion>(x) * static_cast<Promotion>(y);
    } else {
      is_valid &= CheckedMulImpl(static_cast<Promotion>(x),
                                 static_cast<Promotion>(y), &presult);
    }
    *result = static_cast<V>(presult);
    return is_valid && IsValueInRangeForNumericType<V>(presult);
  }
};

// Avoid poluting the namespace once we're done with the macro.
#undef USE_OVERFLOW_BUILTINS

// Division just requires a check for a zero denominator or an invalid negation
// on signed min/-1.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool CheckedDivImpl(T x, T y, T* result) {
  if (y && (!std::is_signed<T>::value ||
            x != std::numeric_limits<T>::lowest() || y != static_cast<T>(-1))) {
    *result = x / y;
    return true;
  }
  return false;
}

template <typename T, typename U, class Enable = void>
struct CheckedDivOp {};

template <typename T, typename U>
struct CheckedDivOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V>
  static bool Do(T x, U y, V* result) {
    using Promotion = typename BigEnoughPromotion<T, U>::type;
    Promotion presult;
    // Fail if either operand is out of range for the promoted type.
    // TODO(jschuh): This could be made to work for a broader range of values.
    bool is_valid = IsValueInRangeForNumericType<Promotion>(x) &&
                    IsValueInRangeForNumericType<Promotion>(y);
    is_valid &= CheckedDivImpl(static_cast<Promotion>(x),
                               static_cast<Promotion>(y), &presult);
    *result = static_cast<V>(presult);
    return is_valid && IsValueInRangeForNumericType<V>(presult);
  }
};

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool CheckedModImpl(T x, T y, T* result) {
  if (y > 0) {
    *result = static_cast<T>(x % y);
    return true;
  }
  return false;
}

template <typename T, typename U, class Enable = void>
struct CheckedModOp {};

template <typename T, typename U>
struct CheckedModOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V>
  static bool Do(T x, U y, V* result) {
    using Promotion = typename BigEnoughPromotion<T, U>::type;
    Promotion presult;
    bool is_valid = CheckedModImpl(static_cast<Promotion>(x),
                                   static_cast<Promotion>(y), &presult);
    *result = static_cast<V>(presult);
    return is_valid && IsValueInRangeForNumericType<V>(presult);
  }
};

template <typename T, typename U, class Enable = void>
struct CheckedLshOp {};

// Left shift. Shifts less than 0 or greater than or equal to the number
// of bits in the promoted type are undefined. Shifts of negative values
// are undefined. Otherwise it is defined when the result fits.
template <typename T, typename U>
struct CheckedLshOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = T;
  template <typename V>
  static bool Do(T x, U shift, V* result) {
    using ShiftType = typename std::make_unsigned<T>::type;
    static const ShiftType kBitWidth = IntegerBitsPlusSign<T>::value;
    const ShiftType real_shift = static_cast<ShiftType>(shift);
    // Signed shift is not legal on negative values.
    if (!IsValueNegative(x) && real_shift < kBitWidth) {
      // Just use a multiplication because it's easy.
      // TODO(jschuh): This could probably be made more efficient.
      if (!std::is_signed<T>::value || real_shift != kBitWidth - 1)
        return CheckedMulOp<T, T>::Do(x, static_cast<T>(1) << shift, result);
      return !x;  // Special case zero for a full width signed shift.
    }
    return false;
  }
};

template <typename T, typename U, class Enable = void>
struct CheckedRshOp {};

// Right shift. Shifts less than 0 or greater than or equal to the number
// of bits in the promoted type are undefined. Otherwise, it is always defined,
// but a right shift of a negative value is implementation-dependent.
template <typename T, typename U>
struct CheckedRshOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = T;
  template <typename V = result_type>
  static bool Do(T x, U shift, V* result) {
    // Use the type conversion push negative values out of range.
    using ShiftType = typename std::make_unsigned<T>::type;
    if (static_cast<ShiftType>(shift) < IntegerBitsPlusSign<T>::value) {
      T tmp = x >> shift;
      *result = static_cast<V>(tmp);
      return IsValueInRangeForNumericType<V>(tmp);
    }
    return false;
  }
};

template <typename T, typename U, class Enable = void>
struct CheckedAndOp {};

// For simplicity we support only unsigned integer results.
template <typename T, typename U>
struct CheckedAndOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename std::make_unsigned<
      typename MaxExponentPromotion<T, U>::type>::type;
  template <typename V = result_type>
  static bool Do(T x, U y, V* result) {
    result_type tmp = static_cast<result_type>(x) & static_cast<result_type>(y);
    *result = static_cast<V>(tmp);
    return IsValueInRangeForNumericType<V>(tmp);
  }
};

template <typename T, typename U, class Enable = void>
struct CheckedOrOp {};

// For simplicity we support only unsigned integers.
template <typename T, typename U>
struct CheckedOrOp<T,
                   U,
                   typename std::enable_if<std::is_integral<T>::value &&
                                           std::is_integral<U>::value>::type> {
  using result_type = typename std::make_unsigned<
      typename MaxExponentPromotion<T, U>::type>::type;
  template <typename V = result_type>
  static bool Do(T x, U y, V* result) {
    result_type tmp = static_cast<result_type>(x) | static_cast<result_type>(y);
    *result = static_cast<V>(tmp);
    return IsValueInRangeForNumericType<V>(tmp);
  }
};

template <typename T, typename U, class Enable = void>
struct CheckedXorOp {};

// For simplicity we support only unsigned integers.
template <typename T, typename U>
struct CheckedXorOp<T,
                    U,
                    typename std::enable_if<std::is_integral<T>::value &&
                                            std::is_integral<U>::value>::type> {
  using result_type = typename std::make_unsigned<
      typename MaxExponentPromotion<T, U>::type>::type;
  template <typename V = result_type>
  static bool Do(T x, U y, V* result) {
    result_type tmp = static_cast<result_type>(x) ^ static_cast<result_type>(y);
    *result = static_cast<V>(tmp);
    return IsValueInRangeForNumericType<V>(tmp);
  }
};

// Max doesn't really need to be implemented this way because it can't fail,
// but it makes the code much cleaner to use the MathOp wrappers.
template <typename T, typename U, class Enable = void>
struct CheckedMaxOp {};

template <typename T, typename U>
struct CheckedMaxOp<
    T,
    U,
    typename std::enable_if<std::is_arithmetic<T>::value &&
                            std::is_arithmetic<U>::value>::type> {
  using result_type = typename MaxExponentPromotion<T, U>::type;
  template <typename V = result_type>
  static bool Do(T x, U y, V* result) {
    *result = IsGreater<T, U>::Test(x, y) ? static_cast<result_type>(x)
                                          : static_cast<result_type>(y);
    return true;
  }
};

// Min doesn't really need to be implemented this way because it can't fail,
// but it makes the code much cleaner to use the MathOp wrappers.
template <typename T, typename U, class Enable = void>
struct CheckedMinOp {};

template <typename T, typename U>
struct CheckedMinOp<
    T,
    U,
    typename std::enable_if<std::is_arithmetic<T>::value &&
                            std::is_arithmetic<U>::value>::type> {
  using result_type = typename LowestValuePromotion<T, U>::type;
  template <typename V = result_type>
  static bool Do(T x, U y, V* result) {
    *result = IsLess<T, U>::Test(x, y) ? static_cast<result_type>(x)
                                       : static_cast<result_type>(y);
    return true;
  }
};

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  std::is_signed<T>::value>::type* = nullptr>
bool CheckedNeg(T value, T* result) {
  // The negation of signed min is min, so catch that one.
  if (value != std::numeric_limits<T>::lowest()) {
    *result = static_cast<T>(-value);
    return true;
  }
  return false;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  !std::is_signed<T>::value>::type* = nullptr>
bool CheckedNeg(T value, T* result) {
  if (!value) {
    *result = static_cast<T>(0);
    return true;
  }
  return false;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  !std::is_signed<T>::value>::type* = nullptr>
bool CheckedInv(T value, T* result) {
  *result = ~value;
  return true;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  std::is_signed<T>::value>::type* = nullptr>
bool CheckedAbs(T value, T* result) {
  if (value != std::numeric_limits<T>::lowest()) {
    *result = std::abs(value);
    return true;
  }
  return false;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  !std::is_signed<T>::value>::type* = nullptr>
bool CheckedAbs(T value, T* result) {
  // T is unsigned, so |value| must already be positive.
  *result = value;
  return true;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  std::is_signed<T>::value>::type* = nullptr>
constexpr typename std::make_unsigned<T>::type SafeUnsignedAbs(T value) {
  using UnsignedT = typename std::make_unsigned<T>::type;
  return value == std::numeric_limits<T>::lowest()
             ? static_cast<UnsignedT>(std::numeric_limits<T>::max()) + 1
             : static_cast<UnsignedT>(std::abs(value));
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                  !std::is_signed<T>::value>::type* = nullptr>
constexpr T SafeUnsignedAbs(T value) {
  // T is unsigned, so |value| must already be positive.
  return static_cast<T>(value);
}

// This is just boilerplate that wraps the standard floating point arithmetic.
// A macro isn't the nicest solution, but it beats rewriting these repeatedly.
#define BASE_FLOAT_ARITHMETIC_OPS(NAME, OP)                                    \
  template <typename T, typename U>                                            \
  struct Checked##NAME##Op<                                                    \
      T, U, typename std::enable_if<std::is_floating_point<T>::value ||        \
                                    std::is_floating_point<U>::value>::type> { \
    using result_type = typename MaxExponentPromotion<T, U>::type;             \
    template <typename V>                                                      \
    static bool Do(T x, U y, V* result) {                                      \
      using Promotion = typename MaxExponentPromotion<T, U>::type;             \
      Promotion presult = x OP y;                                              \
      *result = static_cast<V>(presult);                                       \
      return IsValueInRangeForNumericType<V>(presult);                         \
    }                                                                          \
  };

BASE_FLOAT_ARITHMETIC_OPS(Add, +)
BASE_FLOAT_ARITHMETIC_OPS(Sub, -)
BASE_FLOAT_ARITHMETIC_OPS(Mul, *)
BASE_FLOAT_ARITHMETIC_OPS(Div, /)

#undef BASE_FLOAT_ARITHMETIC_OPS

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
bool CheckedNeg(T value, T* result) {
  *result = static_cast<T>(-value);
  return true;
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
bool CheckedAbs(T value, T* result) {
  *result = static_cast<T>(std::abs(value));
  return true;
}

// Floats carry around their validity state with them, but integers do not. So,
// we wrap the underlying value in a specialization in order to hide that detail
// and expose an interface via accessors.
enum NumericRepresentation {
  NUMERIC_INTEGER,
  NUMERIC_FLOATING,
  NUMERIC_UNKNOWN
};

template <typename NumericType>
struct GetNumericRepresentation {
  static const NumericRepresentation value =
      std::is_integral<NumericType>::value
          ? NUMERIC_INTEGER
          : (std::is_floating_point<NumericType>::value ? NUMERIC_FLOATING
                                                        : NUMERIC_UNKNOWN);
};

template <typename T, NumericRepresentation type =
                          GetNumericRepresentation<T>::value>
class CheckedNumericState {};

// Integrals require quite a bit of additional housekeeping to manage state.
template <typename T>
class CheckedNumericState<T, NUMERIC_INTEGER> {
 private:
  // is_valid_ precedes value_ because member intializers in the constructors
  // are evaluated in field order, and is_valid_ must be read when initializing
  // value_.
  bool is_valid_;
  T value_;

  // Ensures that a type conversion does not trigger undefined behavior.
  template <typename Src>
  static constexpr T WellDefinedConversionOrZero(const Src value,
                                                 const bool is_valid) {
    using SrcType = typename internal::UnderlyingType<Src>::type;
    return (std::is_integral<SrcType>::value || is_valid)
               ? static_cast<T>(value)
               : static_cast<T>(0);
  }

 public:
  template <typename Src, NumericRepresentation type>
  friend class CheckedNumericState;

  constexpr CheckedNumericState() : is_valid_(true), value_(0) {}

  template <typename Src>
  constexpr CheckedNumericState(Src value, bool is_valid)
      : is_valid_(is_valid && IsValueInRangeForNumericType<T>(value)),
        value_(WellDefinedConversionOrZero(value, is_valid_)) {
    static_assert(std::is_arithmetic<Src>::value, "Argument must be numeric.");
  }

  // Copy constructor.
  template <typename Src>
  constexpr CheckedNumericState(const CheckedNumericState<Src>& rhs)
      : is_valid_(rhs.IsValid()),
        value_(WellDefinedConversionOrZero(rhs.value(), is_valid_)) {}

  template <typename Src>
  constexpr explicit CheckedNumericState(Src value)
      : is_valid_(IsValueInRangeForNumericType<T>(value)),
        value_(WellDefinedConversionOrZero(value, is_valid_)) {}

  constexpr bool is_valid() const { return is_valid_; }
  constexpr T value() const { return value_; }
};

// Floating points maintain their own validity, but need translation wrappers.
template <typename T>
class CheckedNumericState<T, NUMERIC_FLOATING> {
 private:
  T value_;

  // Ensures that a type conversion does not trigger undefined behavior.
  template <typename Src>
  static constexpr T WellDefinedConversionOrNaN(const Src value,
                                                const bool is_valid) {
    using SrcType = typename internal::UnderlyingType<Src>::type;
    return (StaticDstRangeRelationToSrcRange<T, SrcType>::value ==
                NUMERIC_RANGE_CONTAINED ||
            is_valid)
               ? static_cast<T>(value)
               : std::numeric_limits<T>::quiet_NaN();
  }

 public:
  template <typename Src, NumericRepresentation type>
  friend class CheckedNumericState;

  constexpr CheckedNumericState() : value_(0.0) {}

  template <typename Src>
  constexpr CheckedNumericState(Src value, bool is_valid)
      : value_(WellDefinedConversionOrNaN(value, is_valid)) {}

  template <typename Src>
  constexpr explicit CheckedNumericState(Src value)
      : value_(WellDefinedConversionOrNaN(
            value,
            IsValueInRangeForNumericType<T>(value))) {}

  // Copy constructor.
  template <typename Src>
  constexpr CheckedNumericState(const CheckedNumericState<Src>& rhs)
      : value_(WellDefinedConversionOrNaN(
            rhs.value(),
            rhs.is_valid() && IsValueInRangeForNumericType<T>(rhs.value()))) {}

  constexpr bool is_valid() const {
    // Written this way because std::isfinite is not reliably constexpr.
    // TODO(jschuh): Fix this if the libraries ever get fixed.
    return value_ <= std::numeric_limits<T>::max() &&
           value_ >= std::numeric_limits<T>::lowest();
  }
  constexpr T value() const { return value_; }
};

template <template <typename, typename, typename> class M,
          typename L,
          typename R>
struct MathWrapper {
  using math = M<typename UnderlyingType<L>::type,
                 typename UnderlyingType<R>::type,
                 void>;
  using type = typename math::result_type;
};

}  // namespace internal
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_IMPL_H_
