// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_H_
#define MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_H_

#include <stddef.h>

#include <limits>
#include <type_traits>

#include "base/numerics/safe_math_impl.h"

namespace base {
namespace internal {

// CheckedNumeric<> implements all the logic and operators for detecting integer
// boundary conditions such as overflow, underflow, and invalid conversions.
// The CheckedNumeric type implicitly converts from floating point and integer
// data types, and contains overloads for basic arithmetic operations (i.e.: +,
// -, *, / for all types and %, <<, >>, &, |, ^ for integers). Type promotions
// are a slightly modified version of the standard C arithmetic rules with the
// two differences being that there is no default promotion to int and bitwise
// logical operations always return an unsigned of the wider type.
//
// You may also use one of the variadic convenience functions, which accept
// standard arithmetic or CheckedNumeric types, perform arithmetic operations,
// and return a CheckedNumeric result. The supported functions are:
//  CheckAdd() - Addition.
//  CheckSub() - Subtraction.
//  CheckMul() - Multiplication.
//  CheckDiv() - Division.
//  CheckMod() - Modulous (integer only).
//  CheckLsh() - Left integer shift (integer only).
//  CheckRsh() - Right integer shift (integer only).
//  CheckAnd() - Bitwise AND (integer only with unsigned result).
//  CheckOr()  - Bitwise OR (integer only with unsigned result).
//  CheckXor() - Bitwise XOR (integer only with unsigned result).
//  CheckMax() - Maximum of supplied arguments.
//  CheckMin() - Minimum of supplied arguments.
//
// The unary negation, increment, and decrement operators are supported, along
// with the following unary arithmetic methods, which return a new
// CheckedNumeric as a result of the operation:
//  Abs() - Absolute value.
//  UnsignedAbs() - Absolute value as an equal-width unsigned underlying type
//          (valid for only integral types).
//  Max() - Returns whichever is greater of the current instance or argument.
//          The underlying return type is whichever has the greatest magnitude.
//  Min() - Returns whichever is lowest of the current instance or argument.
//          The underlying return type is whichever has can represent the lowest
//          number in the smallest width (e.g. int8_t over unsigned, int over
//          int8_t, and float over int).
//
// The following methods convert from CheckedNumeric to standard numeric values:
//  AssignIfValid() - Assigns the underlying value to the supplied destination
//          pointer if the value is currently valid and within the range
//          supported by the destination type. Returns true on success.
//  ****************************************************************************
//  *  WARNING: All of the following functions return a StrictNumeric, which   *
//  *  is valid for comparison and assignment operations, but will trigger a   *
//  *  compile failure on attempts to assign to a type of insufficient range.  *
//  ****************************************************************************
//  IsValid() - Returns true if the underlying numeric value is valid (i.e. has
//          has not wrapped and is not the result of an invalid conversion).
//  ValueOrDie() - Returns the underlying value. If the state is not valid this
//          call will crash on a CHECK.
//  ValueOrDefault() - Returns the current value, or the supplied default if the
//          state is not valid (will not trigger a CHECK).
//
// The following wrapper functions can be used to avoid the template
// disambiguator syntax when converting a destination type.
//   IsValidForType<>() in place of: a.template IsValid<Dst>()
//   ValueOrDieForType<>() in place of: a.template ValueOrDie()
//   ValueOrDefaultForType<>() in place of: a.template ValueOrDefault(default)
//
// The following are general utility methods that are useful for converting
// between arithmetic types and CheckedNumeric types:
//  CheckedNumeric::Cast<Dst>() - Instance method returning a CheckedNumeric
//          derived from casting the current instance to a CheckedNumeric of
//          the supplied destination type.
//  MakeCheckedNum() - Creates a new CheckedNumeric from the underlying type of
//          the supplied arithmetic, CheckedNumeric, or StrictNumeric type.
//
// Comparison operations are explicitly not supported because they could result
// in a crash on an unexpected CHECK condition. You should use patterns like the
// following for comparisons:
//   CheckedNumeric<size_t> checked_size = untrusted_input_value;
//   checked_size += HEADER LENGTH;
//   if (checked_size.IsValid() && checked_size.ValueOrDie() < buffer_size)
//     Do stuff...

template <typename T>
class CheckedNumeric {
  static_assert(std::is_arithmetic<T>::value,
                "CheckedNumeric<T>: T must be a numeric type.");

 public:
  using type = T;

  constexpr CheckedNumeric() {}

  // Copy constructor.
  template <typename Src>
  constexpr CheckedNumeric(const CheckedNumeric<Src>& rhs)
      : state_(rhs.state_.value(), rhs.IsValid()) {}

  template <typename Src>
  friend class CheckedNumeric;

  // This is not an explicit constructor because we implicitly upgrade regular
  // numerics to CheckedNumerics to make them easier to use.
  template <typename Src>
  constexpr CheckedNumeric(Src value)  // NOLINT(runtime/explicit)
      : state_(value) {
    static_assert(std::is_arithmetic<Src>::value, "Argument must be numeric.");
  }

  // This is not an explicit constructor because we want a seamless conversion
  // from StrictNumeric types.
  template <typename Src>
  constexpr CheckedNumeric(
      StrictNumeric<Src> value)  // NOLINT(runtime/explicit)
      : state_(static_cast<Src>(value)) {}

  // IsValid() - The public API to test if a CheckedNumeric is currently valid.
  // A range checked destination type can be supplied using the Dst template
  // parameter.
  template <typename Dst = T>
  constexpr bool IsValid() const {
    return state_.is_valid() &&
           IsValueInRangeForNumericType<Dst>(state_.value());
  }

  // AssignIfValid(Dst) - Assigns the underlying value if it is currently valid
  // and is within the range supported by the destination type. Returns true if
  // successful and false otherwise.
  template <typename Dst>
  constexpr bool AssignIfValid(Dst* result) const {
    return IsValid<Dst>() ? ((*result = static_cast<Dst>(state_.value())), true)
                          : false;
  }

  // ValueOrDie() - The primary accessor for the underlying value. If the
  // current state is not valid it will CHECK and crash.
  // A range checked destination type can be supplied using the Dst template
  // parameter, which will trigger a CHECK if the value is not in bounds for
  // the destination.
  // The CHECK behavior can be overridden by supplying a handler as a
  // template parameter, for test code, etc. However, the handler cannot access
  // the underlying value, and it is not available through other means.
  template <typename Dst = T, class CheckHandler = CheckOnFailure>
  constexpr StrictNumeric<Dst> ValueOrDie() const {
    return IsValid<Dst>() ? static_cast<Dst>(state_.value())
                          : CheckHandler::template HandleFailure<Dst>();
  }

  // ValueOrDefault(T default_value) - A convenience method that returns the
  // current value if the state is valid, and the supplied default_value for
  // any other state.
  // A range checked destination type can be supplied using the Dst template
  // parameter. WARNING: This function may fail to compile or CHECK at runtime
  // if the supplied default_value is not within range of the destination type.
  template <typename Dst = T, typename Src>
  constexpr StrictNumeric<Dst> ValueOrDefault(const Src default_value) const {
    return IsValid<Dst>() ? static_cast<Dst>(state_.value())
                          : checked_cast<Dst>(default_value);
  }

  // Returns a checked numeric of the specified type, cast from the current
  // CheckedNumeric. If the current state is invalid or the destination cannot
  // represent the result then the returned CheckedNumeric will be invalid.
  template <typename Dst>
  constexpr CheckedNumeric<typename UnderlyingType<Dst>::type> Cast() const {
    return *this;
  }

  // This friend method is available solely for providing more detailed logging
  // in the the tests. Do not implement it in production code, because the
  // underlying values may change at any time.
  template <typename U>
  friend U GetNumericValueForTest(const CheckedNumeric<U>& src);

  // Prototypes for the supported arithmetic operator overloads.
  template <typename Src>
  CheckedNumeric& operator+=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator-=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator*=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator/=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator%=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator<<=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator>>=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator&=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator|=(const Src rhs);
  template <typename Src>
  CheckedNumeric& operator^=(const Src rhs);

  CheckedNumeric operator-() const {
    // Negation is always valid for floating point.
    T value = 0;
    bool is_valid = (std::is_floating_point<T>::value || IsValid()) &&
                    CheckedNeg(state_.value(), &value);
    return CheckedNumeric<T>(value, is_valid);
  }

  CheckedNumeric operator~() const {
    static_assert(!std::is_signed<T>::value, "Type must be unsigned.");
    T value = 0;
    bool is_valid = IsValid() && CheckedInv(state_.value(), &value);
    return CheckedNumeric<T>(value, is_valid);
  }

  CheckedNumeric Abs() const {
    // Absolute value is always valid for floating point.
    T value = 0;
    bool is_valid = (std::is_floating_point<T>::value || IsValid()) &&
                    CheckedAbs(state_.value(), &value);
    return CheckedNumeric<T>(value, is_valid);
  }

  template <typename U>
  constexpr CheckedNumeric<typename MathWrapper<CheckedMaxOp, T, U>::type> Max(
      const U rhs) const {
    using R = typename UnderlyingType<U>::type;
    using result_type = typename MathWrapper<CheckedMaxOp, T, U>::type;
    // TODO(jschuh): This can be converted to the MathOp version and remain
    // constexpr once we have C++14 support.
    return CheckedNumeric<result_type>(
        static_cast<result_type>(
            IsGreater<T, R>::Test(state_.value(), Wrapper<U>::value(rhs))
                ? state_.value()
                : Wrapper<U>::value(rhs)),
        state_.is_valid() && Wrapper<U>::is_valid(rhs));
  }

  template <typename U>
  constexpr CheckedNumeric<typename MathWrapper<CheckedMinOp, T, U>::type> Min(
      const U rhs) const {
    using R = typename UnderlyingType<U>::type;
    using result_type = typename MathWrapper<CheckedMinOp, T, U>::type;
    // TODO(jschuh): This can be converted to the MathOp version and remain
    // constexpr once we have C++14 support.
    return CheckedNumeric<result_type>(
        static_cast<result_type>(
            IsLess<T, R>::Test(state_.value(), Wrapper<U>::value(rhs))
                ? state_.value()
                : Wrapper<U>::value(rhs)),
        state_.is_valid() && Wrapper<U>::is_valid(rhs));
  }

  // This function is available only for integral types. It returns an unsigned
  // integer of the same width as the source type, containing the absolute value
  // of the source, and properly handling signed min.
  constexpr CheckedNumeric<typename UnsignedOrFloatForSize<T>::type>
  UnsignedAbs() const {
    return CheckedNumeric<typename UnsignedOrFloatForSize<T>::type>(
        SafeUnsignedAbs(state_.value()), state_.is_valid());
  }

  CheckedNumeric& operator++() {
    *this += 1;
    return *this;
  }

  CheckedNumeric operator++(int) {
    CheckedNumeric value = *this;
    *this += 1;
    return value;
  }

  CheckedNumeric& operator--() {
    *this -= 1;
    return *this;
  }

  CheckedNumeric operator--(int) {
    CheckedNumeric value = *this;
    *this -= 1;
    return value;
  }

  // These perform the actual math operations on the CheckedNumerics.
  // Binary arithmetic operations.
  template <template <typename, typename, typename> class M,
            typename L,
            typename R>
  static CheckedNumeric MathOp(const L lhs, const R rhs) {
    using Math = typename MathWrapper<M, L, R>::math;
    T result = 0;
    bool is_valid =
        Wrapper<L>::is_valid(lhs) && Wrapper<R>::is_valid(rhs) &&
        Math::Do(Wrapper<L>::value(lhs), Wrapper<R>::value(rhs), &result);
    return CheckedNumeric<T>(result, is_valid);
  };

  // Assignment arithmetic operations.
  template <template <typename, typename, typename> class M, typename R>
  CheckedNumeric& MathOp(const R rhs) {
    using Math = typename MathWrapper<M, T, R>::math;
    T result = 0;  // Using T as the destination saves a range check.
    bool is_valid = state_.is_valid() && Wrapper<R>::is_valid(rhs) &&
                    Math::Do(state_.value(), Wrapper<R>::value(rhs), &result);
    *this = CheckedNumeric<T>(result, is_valid);
    return *this;
  };

 private:
  CheckedNumericState<T> state_;

  template <typename Src>
  constexpr CheckedNumeric(Src value, bool is_valid)
      : state_(value, is_valid) {}

  // These wrappers allow us to handle state the same way for both
  // CheckedNumeric and POD arithmetic types.
  template <typename Src>
  struct Wrapper {
    static constexpr bool is_valid(Src) { return true; }
    static constexpr Src value(Src value) { return value; }
  };

  template <typename Src>
  struct Wrapper<CheckedNumeric<Src>> {
    static constexpr bool is_valid(const CheckedNumeric<Src> v) {
      return v.IsValid();
    }
    static constexpr Src value(const CheckedNumeric<Src> v) {
      return v.state_.value();
    }
  };

  template <typename Src>
  struct Wrapper<StrictNumeric<Src>> {
    static constexpr bool is_valid(const StrictNumeric<Src>) { return true; }
    static constexpr Src value(const StrictNumeric<Src> v) {
      return static_cast<Src>(v);
    }
  };
};

// Convenience functions to avoid the ugly template disambiguator syntax.
template <typename Dst, typename Src>
constexpr bool IsValidForType(const CheckedNumeric<Src> value) {
  return value.template IsValid<Dst>();
}

template <typename Dst, typename Src>
constexpr StrictNumeric<Dst> ValueOrDieForType(
    const CheckedNumeric<Src> value) {
  return value.template ValueOrDie<Dst>();
}

template <typename Dst, typename Src, typename Default>
constexpr StrictNumeric<Dst> ValueOrDefaultForType(
    const CheckedNumeric<Src> value,
    const Default default_value) {
  return value.template ValueOrDefault<Dst>(default_value);
}

// These variadic templates work out the return types.
// TODO(jschuh): Rip all this out once we have C++14 non-trailing auto support.
template <template <typename, typename, typename> class M,
          typename L,
          typename R,
          typename... Args>
struct ResultType;

template <template <typename, typename, typename> class M,
          typename L,
          typename R>
struct ResultType<M, L, R> {
  using type = typename MathWrapper<M, L, R>::type;
};

template <template <typename, typename, typename> class M,
          typename L,
          typename R,
          typename... Args>
struct ResultType {
  using type =
      typename ResultType<M, typename ResultType<M, L, R>::type, Args...>::type;
};

// Convience wrapper to return a new CheckedNumeric from the provided arithmetic
// or CheckedNumericType.
template <typename T>
constexpr CheckedNumeric<typename UnderlyingType<T>::type> MakeCheckedNum(
    const T value) {
  return value;
}

// These implement the variadic wrapper for the math operations.
template <template <typename, typename, typename> class M,
          typename L,
          typename R>
CheckedNumeric<typename MathWrapper<M, L, R>::type> ChkMathOp(const L lhs,
                                                              const R rhs) {
  using Math = typename MathWrapper<M, L, R>::math;
  return CheckedNumeric<typename Math::result_type>::template MathOp<M>(lhs,
                                                                        rhs);
}

// General purpose wrapper template for arithmetic operations.
template <template <typename, typename, typename> class M,
          typename L,
          typename R,
          typename... Args>
CheckedNumeric<typename ResultType<M, L, R, Args...>::type>
ChkMathOp(const L lhs, const R rhs, const Args... args) {
  auto tmp = ChkMathOp<M>(lhs, rhs);
  return tmp.IsValid() ? ChkMathOp<M>(tmp, args...)
                       : decltype(ChkMathOp<M>(tmp, args...))(tmp);
};

// The following macros are just boilerplate for the standard arithmetic
// operator overloads and variadic function templates. A macro isn't the nicest
// solution, but it beats rewriting these over and over again.
#define BASE_NUMERIC_ARITHMETIC_VARIADIC(NAME)                                \
  template <typename L, typename R, typename... Args>                         \
  CheckedNumeric<typename ResultType<Checked##NAME##Op, L, R, Args...>::type> \
      Check##NAME(const L lhs, const R rhs, const Args... args) {             \
    return ChkMathOp<Checked##NAME##Op, L, R, Args...>(lhs, rhs, args...);    \
  }

#define BASE_NUMERIC_ARITHMETIC_OPERATORS(NAME, OP, COMPOUND_OP)               \
  /* Binary arithmetic operator for all CheckedNumeric operations. */          \
  template <typename L, typename R,                                            \
            typename std::enable_if<IsCheckedOp<L, R>::value>::type* =         \
                nullptr>                                                       \
  CheckedNumeric<typename MathWrapper<Checked##NAME##Op, L, R>::type>          \
  operator OP(const L lhs, const R rhs) {                                      \
    return decltype(lhs OP rhs)::template MathOp<Checked##NAME##Op>(lhs, rhs); \
  }                                                                            \
  /* Assignment arithmetic operator implementation from CheckedNumeric. */     \
  template <typename L>                                                        \
  template <typename R>                                                        \
  CheckedNumeric<L>& CheckedNumeric<L>::operator COMPOUND_OP(const R rhs) {    \
    return MathOp<Checked##NAME##Op>(rhs);                                     \
  }                                                                            \
  /* Variadic arithmetic functions that return CheckedNumeric. */              \
  BASE_NUMERIC_ARITHMETIC_VARIADIC(NAME)

BASE_NUMERIC_ARITHMETIC_OPERATORS(Add, +, +=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Sub, -, -=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Mul, *, *=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Div, /, /=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Mod, %, %=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Lsh, <<, <<=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Rsh, >>, >>=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(And, &, &=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Or, |, |=)
BASE_NUMERIC_ARITHMETIC_OPERATORS(Xor, ^, ^=)
BASE_NUMERIC_ARITHMETIC_VARIADIC(Max)
BASE_NUMERIC_ARITHMETIC_VARIADIC(Min)

#undef BASE_NUMERIC_ARITHMETIC_VARIADIC
#undef BASE_NUMERIC_ARITHMETIC_OPERATORS

// These are some extra StrictNumeric operators to support simple pointer
// arithmetic with our result types. Since wrapping on a pointer is always
// bad, we trigger the CHECK condition here.
template <typename L, typename R>
L* operator+(L* lhs, const StrictNumeric<R> rhs) {
  uintptr_t result = CheckAdd(reinterpret_cast<uintptr_t>(lhs),
                              CheckMul(sizeof(L), static_cast<R>(rhs)))
                         .template ValueOrDie<uintptr_t>();
  return reinterpret_cast<L*>(result);
}

template <typename L, typename R>
L* operator-(L* lhs, const StrictNumeric<R> rhs) {
  uintptr_t result = CheckSub(reinterpret_cast<uintptr_t>(lhs),
                              CheckMul(sizeof(L), static_cast<R>(rhs)))
                         .template ValueOrDie<uintptr_t>();
  return reinterpret_cast<L*>(result);
}

}  // namespace internal

using internal::CheckedNumeric;
using internal::IsValidForType;
using internal::ValueOrDieForType;
using internal::ValueOrDefaultForType;
using internal::MakeCheckedNum;
using internal::CheckMax;
using internal::CheckMin;
using internal::CheckAdd;
using internal::CheckSub;
using internal::CheckMul;
using internal::CheckDiv;
using internal::CheckMod;
using internal::CheckLsh;
using internal::CheckRsh;
using internal::CheckAnd;
using internal::CheckOr;
using internal::CheckXor;

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_NUMERICS_SAFE_MATH_H_
