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

#ifndef LIBSPIRV_UTIL_HEX_FLOAT_H_
#define LIBSPIRV_UTIL_HEX_FLOAT_H_

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include "bitutils.h"

namespace spvutils {

class Float16 {
 public:
  Float16(uint16_t v) : val(v) {}
  Float16() {}
  static bool isNan(const Float16& val) {
    return ((val.val & 0x7C00) == 0x7C00) && ((val.val & 0x3FF) != 0);
  }
  // Returns true if the given value is any kind of infinity.
  static bool isInfinity(const Float16& val) {
    return ((val.val & 0x7C00) == 0x7C00) && ((val.val & 0x3FF) == 0);
  }
  Float16(const Float16& other) { val = other.val; }
  uint16_t get_value() const { return val; }

  // Returns the maximum normal value.
  static Float16 max() { return Float16(0x7bff); }
  // Returns the lowest normal value.
  static Float16 lowest() { return Float16(0xfbff); }

 private:
  uint16_t val;
};

// To specialize this type, you must override uint_type to define
// an unsigned integer that can fit your floating point type.
// You must also add a isNan function that returns true if
// a value is Nan.
template <typename T>
struct FloatProxyTraits {
  typedef void uint_type;
};

template <>
struct FloatProxyTraits<float> {
  typedef uint32_t uint_type;
  static bool isNan(float f) { return std::isnan(f); }
  // Returns true if the given value is any kind of infinity.
  static bool isInfinity(float f) { return std::isinf(f); }
  // Returns the maximum normal value.
  static float max() { return std::numeric_limits<float>::max(); }
  // Returns the lowest normal value.
  static float lowest() { return std::numeric_limits<float>::lowest(); }
};

template <>
struct FloatProxyTraits<double> {
  typedef uint64_t uint_type;
  static bool isNan(double f) { return std::isnan(f); }
  // Returns true if the given value is any kind of infinity.
  static bool isInfinity(double f) { return std::isinf(f); }
  // Returns the maximum normal value.
  static double max() { return std::numeric_limits<double>::max(); }
  // Returns the lowest normal value.
  static double lowest() { return std::numeric_limits<double>::lowest(); }
};

template <>
struct FloatProxyTraits<Float16> {
  typedef uint16_t uint_type;
  static bool isNan(Float16 f) { return Float16::isNan(f); }
  // Returns true if the given value is any kind of infinity.
  static bool isInfinity(Float16 f) { return Float16::isInfinity(f); }
  // Returns the maximum normal value.
  static Float16 max() { return Float16::max(); }
  // Returns the lowest normal value.
  static Float16 lowest() { return Float16::lowest(); }
};

// Since copying a floating point number (especially if it is NaN)
// does not guarantee that bits are preserved, this class lets us
// store the type and use it as a float when necessary.
template <typename T>
class FloatProxy {
 public:
  typedef typename FloatProxyTraits<T>::uint_type uint_type;

  // Since this is to act similar to the normal floats,
  // do not initialize the data by default.
  FloatProxy() {}

  // Intentionally non-explicit. This is a proxy type so
  // implicit conversions allow us to use it more transparently.
  FloatProxy(T val) { data_ = BitwiseCast<uint_type>(val); }

  // Intentionally non-explicit. This is a proxy type so
  // implicit conversions allow us to use it more transparently.
  FloatProxy(uint_type val) { data_ = val; }

  // This is helpful to have and is guaranteed not to stomp bits.
  FloatProxy<T> operator-() const {
    return static_cast<uint_type>(data_ ^
                                  (uint_type(0x1) << (sizeof(T) * 8 - 1)));
  }

  // Returns the data as a floating point value.
  T getAsFloat() const { return BitwiseCast<T>(data_); }

  // Returns the raw data.
  uint_type data() const { return data_; }

  // Returns true if the value represents any type of NaN.
  bool isNan() { return FloatProxyTraits<T>::isNan(getAsFloat()); }
  // Returns true if the value represents any type of infinity.
  bool isInfinity() { return FloatProxyTraits<T>::isInfinity(getAsFloat()); }

  // Returns the maximum normal value.
  static FloatProxy<T> max() {
    return FloatProxy<T>(FloatProxyTraits<T>::max());
  }
  // Returns the lowest normal value.
  static FloatProxy<T> lowest() {
    return FloatProxy<T>(FloatProxyTraits<T>::lowest());
  }

 private:
  uint_type data_;
};

template <typename T>
bool operator==(const FloatProxy<T>& first, const FloatProxy<T>& second) {
  return first.data() == second.data();
}

// Reads a FloatProxy value as a normal float from a stream.
template <typename T>
std::istream& operator>>(std::istream& is, FloatProxy<T>& value) {
  T float_val;
  is >> float_val;
  value = FloatProxy<T>(float_val);
  return is;
}

// This is an example traits. It is not meant to be used in practice, but will
// be the default for any non-specialized type.
template <typename T>
struct HexFloatTraits {
  // Integer type that can store this hex-float.
  typedef void uint_type;
  // Signed integer type that can store this hex-float.
  typedef void int_type;
  // The numerical type that this HexFloat represents.
  typedef void underlying_type;
  // The type needed to construct the underlying type.
  typedef void native_type;
  // The number of bits that are actually relevant in the uint_type.
  // This allows us to deal with, for example, 24-bit values in a 32-bit
  // integer.
  static const uint32_t num_used_bits = 0;
  // Number of bits that represent the exponent.
  static const uint32_t num_exponent_bits = 0;
  // Number of bits that represent the fractional part.
  static const uint32_t num_fraction_bits = 0;
  // The bias of the exponent. (How much we need to subtract from the stored
  // value to get the correct value.)
  static const uint32_t exponent_bias = 0;
};

// Traits for IEEE float.
// 1 sign bit, 8 exponent bits, 23 fractional bits.
template <>
struct HexFloatTraits<FloatProxy<float>> {
  typedef uint32_t uint_type;
  typedef int32_t int_type;
  typedef FloatProxy<float> underlying_type;
  typedef float native_type;
  static const uint_type num_used_bits = 32;
  static const uint_type num_exponent_bits = 8;
  static const uint_type num_fraction_bits = 23;
  static const uint_type exponent_bias = 127;
};

// Traits for IEEE double.
// 1 sign bit, 11 exponent bits, 52 fractional bits.
template <>
struct HexFloatTraits<FloatProxy<double>> {
  typedef uint64_t uint_type;
  typedef int64_t int_type;
  typedef FloatProxy<double> underlying_type;
  typedef double native_type;
  static const uint_type num_used_bits = 64;
  static const uint_type num_exponent_bits = 11;
  static const uint_type num_fraction_bits = 52;
  static const uint_type exponent_bias = 1023;
};

// Traits for IEEE half.
// 1 sign bit, 5 exponent bits, 10 fractional bits.
template <>
struct HexFloatTraits<FloatProxy<Float16>> {
  typedef uint16_t uint_type;
  typedef int16_t int_type;
  typedef uint16_t underlying_type;
  typedef uint16_t native_type;
  static const uint_type num_used_bits = 16;
  static const uint_type num_exponent_bits = 5;
  static const uint_type num_fraction_bits = 10;
  static const uint_type exponent_bias = 15;
};

enum round_direction {
  kRoundToZero,
  kRoundToNearestEven,
  kRoundToPositiveInfinity,
  kRoundToNegativeInfinity
};

// Template class that houses a floating pointer number.
// It exposes a number of constants based on the provided traits to
// assist in interpreting the bits of the value.
template <typename T, typename Traits = HexFloatTraits<T>>
class HexFloat {
 public:
  typedef typename Traits::uint_type uint_type;
  typedef typename Traits::int_type int_type;
  typedef typename Traits::underlying_type underlying_type;
  typedef typename Traits::native_type native_type;

  explicit HexFloat(T f) : value_(f) {}

  T value() const { return value_; }
  void set_value(T f) { value_ = f; }

  // These are all written like this because it is convenient to have
  // compile-time constants for all of these values.

  // Pass-through values to save typing.
  static const uint32_t num_used_bits = Traits::num_used_bits;
  static const uint32_t exponent_bias = Traits::exponent_bias;
  static const uint32_t num_exponent_bits = Traits::num_exponent_bits;
  static const uint32_t num_fraction_bits = Traits::num_fraction_bits;

  // Number of bits to shift left to set the highest relevant bit.
  static const uint32_t top_bit_left_shift = num_used_bits - 1;
  // How many nibbles (hex characters) the fractional part takes up.
  static const uint32_t fraction_nibbles = (num_fraction_bits + 3) / 4;
  // If the fractional part does not fit evenly into a hex character (4-bits)
  // then we have to left-shift to get rid of leading 0s. This is the amount
  // we have to shift (might be 0).
  static const uint32_t num_overflow_bits =
      fraction_nibbles * 4 - num_fraction_bits;

  // The representation of the fraction, not the actual bits. This
  // includes the leading bit that is usually implicit.
  static const uint_type fraction_represent_mask =
      spvutils::SetBits<uint_type, 0,
                        num_fraction_bits + num_overflow_bits>::get;

  // The topmost bit in the nibble-aligned fraction.
  static const uint_type fraction_top_bit =
      uint_type(1) << (num_fraction_bits + num_overflow_bits - 1);

  // The least significant bit in the exponent, which is also the bit
  // immediately to the left of the significand.
  static const uint_type first_exponent_bit = uint_type(1)
                                              << (num_fraction_bits);

  // The mask for the encoded fraction. It does not include the
  // implicit bit.
  static const uint_type fraction_encode_mask =
      spvutils::SetBits<uint_type, 0, num_fraction_bits>::get;

  // The bit that is used as a sign.
  static const uint_type sign_mask = uint_type(1) << top_bit_left_shift;

  // The bits that represent the exponent.
  static const uint_type exponent_mask =
      spvutils::SetBits<uint_type, num_fraction_bits, num_exponent_bits>::get;

  // How far left the exponent is shifted.
  static const uint32_t exponent_left_shift = num_fraction_bits;

  // How far from the right edge the fraction is shifted.
  static const uint32_t fraction_right_shift =
      static_cast<uint32_t>(sizeof(uint_type) * 8) - num_fraction_bits;

  // The maximum representable unbiased exponent.
  static const int_type max_exponent =
      (exponent_mask >> num_fraction_bits) - exponent_bias;
  // The minimum representable exponent for normalized numbers.
  static const int_type min_exponent = -static_cast<int_type>(exponent_bias);

  // Returns the bits associated with the value.
  uint_type getBits() const { return spvutils::BitwiseCast<uint_type>(value_); }

  // Returns the bits associated with the value, without the leading sign bit.
  uint_type getUnsignedBits() const {
    return static_cast<uint_type>(spvutils::BitwiseCast<uint_type>(value_) &
                                  ~sign_mask);
  }

  // Returns the bits associated with the exponent, shifted to start at the
  // lsb of the type.
  const uint_type getExponentBits() const {
    return static_cast<uint_type>((getBits() & exponent_mask) >>
                                  num_fraction_bits);
  }

  // Returns the exponent in unbiased form. This is the exponent in the
  // human-friendly form.
  const int_type getUnbiasedExponent() const {
    return static_cast<int_type>(getExponentBits() - exponent_bias);
  }

  // Returns just the significand bits from the value.
  const uint_type getSignificandBits() const {
    return getBits() & fraction_encode_mask;
  }

  // If the number was normalized, returns the unbiased exponent.
  // If the number was denormal, normalize the exponent first.
  const int_type getUnbiasedNormalizedExponent() const {
    if ((getBits() & ~sign_mask) == 0) {  // special case if everything is 0
      return 0;
    }
    int_type exp = getUnbiasedExponent();
    if (exp == min_exponent) {  // We are in denorm land.
      uint_type significand_bits = getSignificandBits();
      while ((significand_bits & (first_exponent_bit >> 1)) == 0) {
        significand_bits = static_cast<uint_type>(significand_bits << 1);
        exp = static_cast<int_type>(exp - 1);
      }
      significand_bits &= fraction_encode_mask;
    }
    return exp;
  }

  // Returns the signficand after it has been normalized.
  const uint_type getNormalizedSignificand() const {
    int_type unbiased_exponent = getUnbiasedNormalizedExponent();
    uint_type significand = getSignificandBits();
    for (int_type i = unbiased_exponent; i <= min_exponent; ++i) {
      significand = static_cast<uint_type>(significand << 1);
    }
    significand &= fraction_encode_mask;
    return significand;
  }

  // Returns true if this number represents a negative value.
  bool isNegative() const { return (getBits() & sign_mask) != 0; }

  // Sets this HexFloat from the individual components.
  // Note this assumes EVERY significand is normalized, and has an implicit
  // leading one. This means that the only way that this method will set 0,
  // is if you set a number so denormalized that it underflows.
  // Do not use this method with raw bits extracted from a subnormal number,
  // since subnormals do not have an implicit leading 1 in the significand.
  // The significand is also expected to be in the
  // lowest-most num_fraction_bits of the uint_type.
  // The exponent is expected to be unbiased, meaning an exponent of
  // 0 actually means 0.
  // If underflow_round_up is set, then on underflow, if a number is non-0
  // and would underflow, we round up to the smallest denorm.
  void setFromSignUnbiasedExponentAndNormalizedSignificand(
      bool negative, int_type exponent, uint_type significand,
      bool round_denorm_up) {
    bool significand_is_zero = significand == 0;

    if (exponent <= min_exponent) {
      // If this was denormalized, then we have to shift the bit on, meaning
      // the significand is not zero.
      significand_is_zero = false;
      significand |= first_exponent_bit;
      significand = static_cast<uint_type>(significand >> 1);
    }

    while (exponent < min_exponent) {
      significand = static_cast<uint_type>(significand >> 1);
      ++exponent;
    }

    if (exponent == min_exponent) {
      if (significand == 0 && !significand_is_zero && round_denorm_up) {
        significand = static_cast<uint_type>(0x1);
      }
    }

    uint_type new_value = 0;
    if (negative) {
      new_value = static_cast<uint_type>(new_value | sign_mask);
    }
    exponent = static_cast<int_type>(exponent + exponent_bias);
    assert(exponent >= 0);

    // put it all together
    exponent = static_cast<uint_type>((exponent << exponent_left_shift) &
                                      exponent_mask);
    significand = static_cast<uint_type>(significand & fraction_encode_mask);
    new_value = static_cast<uint_type>(new_value | (exponent | significand));
    value_ = BitwiseCast<T>(new_value);
  }

  // Increments the significand of this number by the given amount.
  // If this would spill the significand into the implicit bit,
  // carry is set to true and the significand is shifted to fit into
  // the correct location, otherwise carry is set to false.
  // All significands and to_increment are assumed to be within the bounds
  // for a valid significand.
  static uint_type incrementSignificand(uint_type significand,
                                        uint_type to_increment, bool* carry) {
    significand = static_cast<uint_type>(significand + to_increment);
    *carry = false;
    if (significand & first_exponent_bit) {
      *carry = true;
      // The implicit 1-bit will have carried, so we should zero-out the
      // top bit and shift back.
      significand = static_cast<uint_type>(significand & ~first_exponent_bit);
      significand = static_cast<uint_type>(significand >> 1);
    }
    return significand;
  }

  // These exist because MSVC throws warnings on negative right-shifts
  // even if they are not going to be executed. Eg:
  // constant_number < 0? 0: constant_number
  // These convert the negative left-shifts into right shifts.

  template <typename int_type>
  uint_type negatable_left_shift(int_type N, uint_type val)
  {
    if(N >= 0)
      return val << N;

    return val >> -N;
  }

  template <typename int_type>
  uint_type negatable_right_shift(int_type N, uint_type val)
  {
    if(N >= 0)
      return val >> N;

    return val << -N;
  }

  // Returns the significand, rounded to fit in a significand in
  // other_T. This is shifted so that the most significant
  // bit of the rounded number lines up with the most significant bit
  // of the returned significand.
  template <typename other_T>
  typename other_T::uint_type getRoundedNormalizedSignificand(
      round_direction dir, bool* carry_bit) {
    typedef typename other_T::uint_type other_uint_type;
    static const int_type num_throwaway_bits =
        static_cast<int_type>(num_fraction_bits) -
        static_cast<int_type>(other_T::num_fraction_bits);

    static const uint_type last_significant_bit =
        (num_throwaway_bits < 0)
            ? 0
            : negatable_left_shift(num_throwaway_bits, 1u);
    static const uint_type first_rounded_bit =
        (num_throwaway_bits < 1)
            ? 0
            : negatable_left_shift(num_throwaway_bits - 1, 1u);

    static const uint_type throwaway_mask_bits =
        num_throwaway_bits > 0 ? num_throwaway_bits : 0;
    static const uint_type throwaway_mask =
        spvutils::SetBits<uint_type, 0, throwaway_mask_bits>::get;

    *carry_bit = false;
    other_uint_type out_val = 0;
    uint_type significand = getNormalizedSignificand();
    // If we are up-casting, then we just have to shift to the right location.
    if (num_throwaway_bits <= 0) {
      out_val = static_cast<other_uint_type>(significand);
      uint_type shift_amount = static_cast<uint_type>(-num_throwaway_bits);
      out_val = static_cast<other_uint_type>(out_val << shift_amount);
      return out_val;
    }

    // If every non-representable bit is 0, then we don't have any casting to
    // do.
    if ((significand & throwaway_mask) == 0) {
      return static_cast<other_uint_type>(
          negatable_right_shift(num_throwaway_bits, significand));
    }

    bool round_away_from_zero = false;
    // We actually have to narrow the significand here, so we have to follow the
    // rounding rules.
    switch (dir) {
      case kRoundToZero:
        break;
      case kRoundToPositiveInfinity:
        round_away_from_zero = !isNegative();
        break;
      case kRoundToNegativeInfinity:
        round_away_from_zero = isNegative();
        break;
      case kRoundToNearestEven:
        // Have to round down, round bit is 0
        if ((first_rounded_bit & significand) == 0) {
          break;
        }
        if (((significand & throwaway_mask) & ~first_rounded_bit) != 0) {
          // If any subsequent bit of the rounded portion is non-0 then we round
          // up.
          round_away_from_zero = true;
          break;
        }
        // We are exactly half-way between 2 numbers, pick even.
        if ((significand & last_significant_bit) != 0) {
          // 1 for our last bit, round up.
          round_away_from_zero = true;
          break;
        }
        break;
    }

    if (round_away_from_zero) {
      return static_cast<other_uint_type>(
          negatable_right_shift(num_throwaway_bits, incrementSignificand(
              significand, last_significant_bit, carry_bit)));
    } else {
      return static_cast<other_uint_type>(
          negatable_right_shift(num_throwaway_bits, significand));
    }
  }

  // Casts this value to another HexFloat. If the cast is widening,
  // then round_dir is ignored. If the cast is narrowing, then
  // the result is rounded in the direction specified.
  // This number will retain Nan and Inf values.
  // It will also saturate to Inf if the number overflows, and
  // underflow to (0 or min depending on rounding) if the number underflows.
  template <typename other_T>
  void castTo(other_T& other, round_direction round_dir) {
    other = other_T(static_cast<typename other_T::native_type>(0));
    bool negate = isNegative();
    if (getUnsignedBits() == 0) {
      if (negate) {
        other.set_value(-other.value());
      }
      return;
    }
    uint_type significand = getSignificandBits();
    bool carried = false;
    typename other_T::uint_type rounded_significand =
        getRoundedNormalizedSignificand<other_T>(round_dir, &carried);

    int_type exponent = getUnbiasedExponent();
    if (exponent == min_exponent) {
      // If we are denormal, normalize the exponent, so that we can encode
      // easily.
      exponent = static_cast<int_type>(exponent + 1);
      for (uint_type check_bit = first_exponent_bit >> 1; check_bit != 0;
           check_bit = static_cast<uint_type>(check_bit >> 1)) {
        exponent = static_cast<int_type>(exponent - 1);
        if (check_bit & significand) break;
      }
    }

    bool is_nan =
        (getBits() & exponent_mask) == exponent_mask && significand != 0;
    bool is_inf =
        !is_nan &&
        ((exponent + carried) > static_cast<int_type>(other_T::exponent_bias) ||
         (significand == 0 && (getBits() & exponent_mask) == exponent_mask));

    // If we are Nan or Inf we should pass that through.
    if (is_inf) {
      other.set_value(BitwiseCast<typename other_T::underlying_type>(
          static_cast<typename other_T::uint_type>(
              (negate ? other_T::sign_mask : 0) | other_T::exponent_mask)));
      return;
    }
    if (is_nan) {
      typename other_T::uint_type shifted_significand;
      shifted_significand = static_cast<typename other_T::uint_type>(
          negatable_left_shift(
              static_cast<int_type>(other_T::num_fraction_bits) -
              static_cast<int_type>(num_fraction_bits), significand));

      // We are some sort of Nan. We try to keep the bit-pattern of the Nan
      // as close as possible. If we had to shift off bits so we are 0, then we
      // just set the last bit.
      other.set_value(BitwiseCast<typename other_T::underlying_type>(
          static_cast<typename other_T::uint_type>(
              (negate ? other_T::sign_mask : 0) | other_T::exponent_mask |
              (shifted_significand == 0 ? 0x1 : shifted_significand))));
      return;
    }

    bool round_underflow_up =
        isNegative() ? round_dir == kRoundToNegativeInfinity
                     : round_dir == kRoundToPositiveInfinity;
    typedef typename other_T::int_type other_int_type;
    // setFromSignUnbiasedExponentAndNormalizedSignificand will
    // zero out any underflowing value (but retain the sign).
    other.setFromSignUnbiasedExponentAndNormalizedSignificand(
        negate, static_cast<other_int_type>(exponent), rounded_significand,
        round_underflow_up);
    return;
  }

 private:
  T value_;

  static_assert(num_used_bits ==
                    Traits::num_exponent_bits + Traits::num_fraction_bits + 1,
                "The number of bits do not fit");
  static_assert(sizeof(T) == sizeof(uint_type), "The type sizes do not match");
};

// Returns 4 bits represented by the hex character.
inline uint8_t get_nibble_from_character(int character) {
  const char* dec = "0123456789";
  const char* lower = "abcdef";
  const char* upper = "ABCDEF";
  const char* p = nullptr;
  if ((p = strchr(dec, character))) {
    return static_cast<uint8_t>(p - dec);
  } else if ((p = strchr(lower, character))) {
    return static_cast<uint8_t>(p - lower + 0xa);
  } else if ((p = strchr(upper, character))) {
    return static_cast<uint8_t>(p - upper + 0xa);
  }

  assert(false && "This was called with a non-hex character");
  return 0;
}

// Outputs the given HexFloat to the stream.
template <typename T, typename Traits>
std::ostream& operator<<(std::ostream& os, const HexFloat<T, Traits>& value) {
  typedef HexFloat<T, Traits> HF;
  typedef typename HF::uint_type uint_type;
  typedef typename HF::int_type int_type;

  static_assert(HF::num_used_bits != 0,
                "num_used_bits must be non-zero for a valid float");
  static_assert(HF::num_exponent_bits != 0,
                "num_exponent_bits must be non-zero for a valid float");
  static_assert(HF::num_fraction_bits != 0,
                "num_fractin_bits must be non-zero for a valid float");

  const uint_type bits = spvutils::BitwiseCast<uint_type>(value.value());
  const char* const sign = (bits & HF::sign_mask) ? "-" : "";
  const uint_type exponent = static_cast<uint_type>(
      (bits & HF::exponent_mask) >> HF::num_fraction_bits);

  uint_type fraction = static_cast<uint_type>((bits & HF::fraction_encode_mask)
                                              << HF::num_overflow_bits);

  const bool is_zero = exponent == 0 && fraction == 0;
  const bool is_denorm = exponent == 0 && !is_zero;

  // exponent contains the biased exponent we have to convert it back into
  // the normal range.
  int_type int_exponent = static_cast<int_type>(exponent - HF::exponent_bias);
  // If the number is all zeros, then we actually have to NOT shift the
  // exponent.
  int_exponent = is_zero ? 0 : int_exponent;

  // If we are denorm, then start shifting, and decreasing the exponent until
  // our leading bit is 1.

  if (is_denorm) {
    while ((fraction & HF::fraction_top_bit) == 0) {
      fraction = static_cast<uint_type>(fraction << 1);
      int_exponent = static_cast<int_type>(int_exponent - 1);
    }
    // Since this is denormalized, we have to consume the leading 1 since it
    // will end up being implicit.
    fraction = static_cast<uint_type>(fraction << 1);  // eat the leading 1
    fraction &= HF::fraction_represent_mask;
  }

  uint_type fraction_nibbles = HF::fraction_nibbles;
  // We do not have to display any trailing 0s, since this represents the
  // fractional part.
  while (fraction_nibbles > 0 && (fraction & 0xF) == 0) {
    // Shift off any trailing values;
    fraction = static_cast<uint_type>(fraction >> 4);
    --fraction_nibbles;
  }

  const auto saved_flags = os.flags();
  const auto saved_fill = os.fill();

  os << sign << "0x" << (is_zero ? '0' : '1');
  if (fraction_nibbles) {
    // Make sure to keep the leading 0s in place, since this is the fractional
    // part.
    os << "." << std::setw(static_cast<int>(fraction_nibbles))
       << std::setfill('0') << std::hex << fraction;
  }
  os << "p" << std::dec << (int_exponent >= 0 ? "+" : "") << int_exponent;

  os.flags(saved_flags);
  os.fill(saved_fill);

  return os;
}

// Returns true if negate_value is true and the next character on the
// input stream is a plus or minus sign.  In that case we also set the fail bit
// on the stream and set the value to the zero value for its type.
template <typename T, typename Traits>
inline bool RejectParseDueToLeadingSign(std::istream& is, bool negate_value,
                                        HexFloat<T, Traits>& value) {
  if (negate_value) {
    auto next_char = is.peek();
    if (next_char == '-' || next_char == '+') {
      // Fail the parse.  Emulate standard behaviour by setting the value to
      // the zero value, and set the fail bit on the stream.
      value = HexFloat<T, Traits>(typename HexFloat<T, Traits>::uint_type(0));
      is.setstate(std::ios_base::failbit);
      return true;
    }
  }
  return false;
}

// Parses a floating point number from the given stream and stores it into the
// value parameter.
// If negate_value is true then the number may not have a leading minus or
// plus, and if it successfully parses, then the number is negated before
// being stored into the value parameter.
// If the value cannot be correctly parsed or overflows the target floating
// point type, then set the fail bit on the stream.
// TODO(dneto): Promise C++11 standard behavior in how the value is set in
// the error case, but only after all target platforms implement it correctly.
// In particular, the Microsoft C++ runtime appears to be out of spec.
template <typename T, typename Traits>
inline std::istream& ParseNormalFloat(std::istream& is, bool negate_value,
                                      HexFloat<T, Traits>& value) {
  if (RejectParseDueToLeadingSign(is, negate_value, value)) {
    return is;
  }
  T val;
  is >> val;
  if (negate_value) {
    val = -val;
  }
  value.set_value(val);
  // In the failure case, map -0.0 to 0.0.
  if (is.fail() && value.getUnsignedBits() == 0u) {
    value = HexFloat<T, Traits>(typename HexFloat<T, Traits>::uint_type(0));
  }
  if (val.isInfinity()) {
    // Fail the parse.  Emulate standard behaviour by setting the value to
    // the closest normal value, and set the fail bit on the stream.
    value.set_value((value.isNegative() || negate_value) ? T::lowest()
                                                         : T::max());
    is.setstate(std::ios_base::failbit);
  }
  return is;
}

// Specialization of ParseNormalFloat for FloatProxy<Float16> values.
// This will parse the float as it were a 32-bit floating point number,
// and then round it down to fit into a Float16 value.
// The number is rounded towards zero.
// If negate_value is true then the number may not have a leading minus or
// plus, and if it successfully parses, then the number is negated before
// being stored into the value parameter.
// If the value cannot be correctly parsed or overflows the target floating
// point type, then set the fail bit on the stream.
// TODO(dneto): Promise C++11 standard behavior in how the value is set in
// the error case, but only after all target platforms implement it correctly.
// In particular, the Microsoft C++ runtime appears to be out of spec.
template <>
inline std::istream&
ParseNormalFloat<FloatProxy<Float16>, HexFloatTraits<FloatProxy<Float16>>>(
    std::istream& is, bool negate_value,
    HexFloat<FloatProxy<Float16>, HexFloatTraits<FloatProxy<Float16>>>& value) {
  // First parse as a 32-bit float.
  HexFloat<FloatProxy<float>> float_val(0.0f);
  ParseNormalFloat(is, negate_value, float_val);

  // Then convert to 16-bit float, saturating at infinities, and
  // rounding toward zero.
  float_val.castTo(value, kRoundToZero);

  // Overflow on 16-bit behaves the same as for 32- and 64-bit: set the
  // fail bit and set the lowest or highest value.
  if (Float16::isInfinity(value.value().getAsFloat())) {
    value.set_value(value.isNegative() ? Float16::lowest() : Float16::max());
    is.setstate(std::ios_base::failbit);
  }
  return is;
}

// Reads a HexFloat from the given stream.
// If the float is not encoded as a hex-float then it will be parsed
// as a regular float.
// This may fail if your stream does not support at least one unget.
// Nan values can be encoded with "0x1.<not zero>p+exponent_bias".
// This would normally overflow a float and round to
// infinity but this special pattern is the exact representation for a NaN,
// and therefore is actually encoded as the correct NaN. To encode inf,
// either 0x0p+exponent_bias can be specified or any exponent greater than
// exponent_bias.
// Examples using IEEE 32-bit float encoding.
//    0x1.0p+128 (+inf)
//    -0x1.0p-128 (-inf)
//
//    0x1.1p+128 (+Nan)
//    -0x1.1p+128 (-Nan)
//
//    0x1p+129 (+inf)
//    -0x1p+129 (-inf)
template <typename T, typename Traits>
std::istream& operator>>(std::istream& is, HexFloat<T, Traits>& value) {
  using HF = HexFloat<T, Traits>;
  using uint_type = typename HF::uint_type;
  using int_type = typename HF::int_type;

  value.set_value(static_cast<typename HF::native_type>(0.f));

  if (is.flags() & std::ios::skipws) {
    // If the user wants to skip whitespace , then we should obey that.
    while (std::isspace(is.peek())) {
      is.get();
    }
  }

  auto next_char = is.peek();
  bool negate_value = false;

  if (next_char != '-' && next_char != '0') {
    return ParseNormalFloat(is, negate_value, value);
  }

  if (next_char == '-') {
    negate_value = true;
    is.get();
    next_char = is.peek();
  }

  if (next_char == '0') {
    is.get();  // We may have to unget this.
    auto maybe_hex_start = is.peek();
    if (maybe_hex_start != 'x' && maybe_hex_start != 'X') {
      is.unget();
      return ParseNormalFloat(is, negate_value, value);
    } else {
      is.get();  // Throw away the 'x';
    }
  } else {
    return ParseNormalFloat(is, negate_value, value);
  }

  // This "looks" like a hex-float so treat it as one.
  bool seen_p = false;
  bool seen_dot = false;
  uint_type fraction_index = 0;

  uint_type fraction = 0;
  int_type exponent = HF::exponent_bias;

  // Strip off leading zeros so we don't have to special-case them later.
  while ((next_char = is.peek()) == '0') {
    is.get();
  }

  bool is_denorm =
      true;  // Assume denorm "representation" until we hear otherwise.
             // NB: This does not mean the value is actually denorm,
             // it just means that it was written 0.
  bool bits_written = false;  // Stays false until we write a bit.
  while (!seen_p && !seen_dot) {
    // Handle characters that are left of the fractional part.
    if (next_char == '.') {
      seen_dot = true;
    } else if (next_char == 'p') {
      seen_p = true;
    } else if (::isxdigit(next_char)) {
      // We know this is not denormalized since we have stripped all leading
      // zeroes and we are not a ".".
      is_denorm = false;
      int number = get_nibble_from_character(next_char);
      for (int i = 0; i < 4; ++i, number <<= 1) {
        uint_type write_bit = (number & 0x8) ? 0x1 : 0x0;
        if (bits_written) {
          // If we are here the bits represented belong in the fractional
          // part of the float, and we have to adjust the exponent accordingly.
          fraction = static_cast<uint_type>(
              fraction |
              static_cast<uint_type>(
                  write_bit << (HF::top_bit_left_shift - fraction_index++)));
          exponent = static_cast<int_type>(exponent + 1);
        }
        bits_written |= write_bit != 0;
      }
    } else {
      // We have not found our exponent yet, so we have to fail.
      is.setstate(std::ios::failbit);
      return is;
    }
    is.get();
    next_char = is.peek();
  }
  bits_written = false;
  while (seen_dot && !seen_p) {
    // Handle only fractional parts now.
    if (next_char == 'p') {
      seen_p = true;
    } else if (::isxdigit(next_char)) {
      int number = get_nibble_from_character(next_char);
      for (int i = 0; i < 4; ++i, number <<= 1) {
        uint_type write_bit = (number & 0x8) ? 0x01 : 0x00;
        bits_written |= write_bit != 0;
        if (is_denorm && !bits_written) {
          // Handle modifying the exponent here this way we can handle
          // an arbitrary number of hex values without overflowing our
          // integer.
          exponent = static_cast<int_type>(exponent - 1);
        } else {
          fraction = static_cast<uint_type>(
              fraction |
              static_cast<uint_type>(
                  write_bit << (HF::top_bit_left_shift - fraction_index++)));
        }
      }
    } else {
      // We still have not found our 'p' exponent yet, so this is not a valid
      // hex-float.
      is.setstate(std::ios::failbit);
      return is;
    }
    is.get();
    next_char = is.peek();
  }

  bool seen_sign = false;
  int8_t exponent_sign = 1;
  int_type written_exponent = 0;
  while (true) {
    if ((next_char == '-' || next_char == '+')) {
      if (seen_sign) {
        is.setstate(std::ios::failbit);
        return is;
      }
      seen_sign = true;
      exponent_sign = (next_char == '-') ? -1 : 1;
    } else if (::isdigit(next_char)) {
      // Hex-floats express their exponent as decimal.
      written_exponent = static_cast<int_type>(written_exponent * 10);
      written_exponent =
          static_cast<int_type>(written_exponent + (next_char - '0'));
    } else {
      break;
    }
    is.get();
    next_char = is.peek();
  }

  written_exponent = static_cast<int_type>(written_exponent * exponent_sign);
  exponent = static_cast<int_type>(exponent + written_exponent);

  bool is_zero = is_denorm && (fraction == 0);
  if (is_denorm && !is_zero) {
    fraction = static_cast<uint_type>(fraction << 1);
    exponent = static_cast<int_type>(exponent - 1);
  } else if (is_zero) {
    exponent = 0;
  }

  if (exponent <= 0 && !is_zero) {
    fraction = static_cast<uint_type>(fraction >> 1);
    fraction |= static_cast<uint_type>(1) << HF::top_bit_left_shift;
  }

  fraction = (fraction >> HF::fraction_right_shift) & HF::fraction_encode_mask;

  const int_type max_exponent =
      SetBits<uint_type, 0, HF::num_exponent_bits>::get;

  // Handle actual denorm numbers
  while (exponent < 0 && !is_zero) {
    fraction = static_cast<uint_type>(fraction >> 1);
    exponent = static_cast<int_type>(exponent + 1);

    fraction &= HF::fraction_encode_mask;
    if (fraction == 0) {
      // We have underflowed our fraction. We should clamp to zero.
      is_zero = true;
      exponent = 0;
    }
  }

  // We have overflowed so we should be inf/-inf.
  if (exponent > max_exponent) {
    exponent = max_exponent;
    fraction = 0;
  }

  uint_type output_bits = static_cast<uint_type>(
      static_cast<uint_type>(negate_value ? 1 : 0) << HF::top_bit_left_shift);
  output_bits |= fraction;

  uint_type shifted_exponent = static_cast<uint_type>(
      static_cast<uint_type>(exponent << HF::exponent_left_shift) &
      HF::exponent_mask);
  output_bits |= shifted_exponent;

  T output_float = spvutils::BitwiseCast<T>(output_bits);
  value.set_value(output_float);

  return is;
}

// Writes a FloatProxy value to a stream.
// Zero and normal numbers are printed in the usual notation, but with
// enough digits to fully reproduce the value.  Other values (subnormal,
// NaN, and infinity) are printed as a hex float.
template <typename T>
std::ostream& operator<<(std::ostream& os, const FloatProxy<T>& value) {
  auto float_val = value.getAsFloat();
  switch (std::fpclassify(float_val)) {
    case FP_ZERO:
    case FP_NORMAL: {
      auto saved_precision = os.precision();
      os.precision(std::numeric_limits<T>::digits10);
      os << float_val;
      os.precision(saved_precision);
    } break;
    default:
      os << HexFloat<FloatProxy<T>>(value);
      break;
  }
  return os;
}

template <>
inline std::ostream& operator<<<Float16>(std::ostream& os,
                                         const FloatProxy<Float16>& value) {
  os << HexFloat<FloatProxy<Float16>>(value);
  return os;
}
}

#endif  // LIBSPIRV_UTIL_HEX_FLOAT_H_
