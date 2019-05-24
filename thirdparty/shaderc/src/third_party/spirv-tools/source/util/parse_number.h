// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_UTIL_PARSE_NUMBER_H_
#define SOURCE_UTIL_PARSE_NUMBER_H_

#include <functional>
#include <string>
#include <tuple>

#include "source/util/hex_float.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace utils {

// A struct to hold the expected type information for the number in text to be
// parsed.
struct NumberType {
  uint32_t bitwidth;
  // SPV_NUMBER_NONE means the type is unknown and is invalid to be used with
  // ParseAndEncode{|Integer|Floating}Number().
  spv_number_kind_t kind;
};

// Returns true if the type is a scalar integer type.
inline bool IsIntegral(const NumberType& type) {
  return type.kind == SPV_NUMBER_UNSIGNED_INT ||
         type.kind == SPV_NUMBER_SIGNED_INT;
}

// Returns true if the type is a scalar floating point type.
inline bool IsFloating(const NumberType& type) {
  return type.kind == SPV_NUMBER_FLOATING;
}

// Returns true if the type is a signed value.
inline bool IsSigned(const NumberType& type) {
  return type.kind == SPV_NUMBER_FLOATING || type.kind == SPV_NUMBER_SIGNED_INT;
}

// Returns true if the type is unknown.
inline bool IsUnknown(const NumberType& type) {
  return type.kind == SPV_NUMBER_NONE;
}

// Returns the number of bits in the type. This is only valid for integer and
// floating types.
inline int AssumedBitWidth(const NumberType& type) {
  switch (type.kind) {
    case SPV_NUMBER_SIGNED_INT:
    case SPV_NUMBER_UNSIGNED_INT:
    case SPV_NUMBER_FLOATING:
      return type.bitwidth;
    default:
      break;
  }
  // We don't care about this case.
  return 0;
}

// A templated class with a static member function Clamp, where Clamp sets a
// referenced value of type T to 0 if T is an unsigned integer type, and
// returns true if it modified the referenced value.
template <typename T, typename = void>
class ClampToZeroIfUnsignedType {
 public:
  // The default specialization does not clamp the value.
  static bool Clamp(T*) { return false; }
};

// The specialization of ClampToZeroIfUnsignedType for unsigned integer types.
template <typename T>
class ClampToZeroIfUnsignedType<
    T, typename std::enable_if<std::is_unsigned<T>::value>::type> {
 public:
  static bool Clamp(T* value_pointer) {
    if (*value_pointer) {
      *value_pointer = 0;
      return true;
    }
    return false;
  }
};

// Returns true if the given value fits within the target scalar integral type.
// The target type may have an unusual bit width. If the value was originally
// specified as a hexadecimal number, then the overflow bits should be zero.
// If it was hex and the target type is signed, then return the sign-extended
// value through the updated_value_for_hex pointer argument. On failure,
// returns false.
template <typename T>
bool CheckRangeAndIfHexThenSignExtend(T value, const NumberType& type,
                                      bool is_hex, T* updated_value_for_hex) {
  // The encoded result has three regions of bits that are of interest, from
  // least to most significant:
  //   - magnitude bits, where the magnitude of the number would be stored if
  //     we were using a signed-magnitude representation.
  //   - an optional sign bit
  //   - overflow bits, up to bit 63 of a 64-bit number
  // For example:
  //   Type                Overflow      Sign       Magnitude
  //   ---------------     --------      ----       ---------
  //   unsigned 8 bit      8-63          n/a        0-7
  //   signed 8 bit        8-63          7          0-6
  //   unsigned 16 bit     16-63         n/a        0-15
  //   signed 16 bit       16-63         15         0-14

  // We'll use masks to define the three regions.
  // At first we'll assume the number is unsigned.
  const uint32_t bit_width = AssumedBitWidth(type);
  uint64_t magnitude_mask =
      (bit_width == 64) ? -1 : ((uint64_t(1) << bit_width) - 1);
  uint64_t sign_mask = 0;
  uint64_t overflow_mask = ~magnitude_mask;

  if (value < 0 || IsSigned(type)) {
    // Accommodate the sign bit.
    magnitude_mask >>= 1;
    sign_mask = magnitude_mask + 1;
  }

  bool failed = false;
  if (value < 0) {
    // The top bits must all be 1 for a negative signed value.
    failed = ((value & overflow_mask) != overflow_mask) ||
             ((value & sign_mask) != sign_mask);
  } else {
    if (is_hex) {
      // Hex values are a bit special. They decode as unsigned values, but may
      // represent a negative number. In this case, the overflow bits should
      // be zero.
      failed = (value & overflow_mask) != 0;
    } else {
      const uint64_t value_as_u64 = static_cast<uint64_t>(value);
      // Check overflow in the ordinary case.
      failed = (value_as_u64 & magnitude_mask) != value_as_u64;
    }
  }

  if (failed) {
    return false;
  }

  // Sign extend hex the number.
  if (is_hex && (value & sign_mask))
    *updated_value_for_hex = (value | overflow_mask);

  return true;
}

// Parses a numeric value of a given type from the given text.  The number
// should take up the entire string, and should be within bounds for the target
// type. On success, returns true and populates the object referenced by
// value_pointer. On failure, returns false.
template <typename T>
bool ParseNumber(const char* text, T* value_pointer) {
  // C++11 doesn't define std::istringstream(int8_t&), so calling this method
  // with a single-byte type leads to implementation-defined behaviour.
  // Similarly for uint8_t.
  static_assert(sizeof(T) > 1,
                "Single-byte types are not supported in this parse method");

  if (!text) return false;
  std::istringstream text_stream(text);
  // Allow both decimal and hex input for integers.
  // It also allows octal input, but we don't care about that case.
  text_stream >> std::setbase(0);
  text_stream >> *value_pointer;

  // We should have read something.
  bool ok = (text[0] != 0) && !text_stream.bad();
  // It should have been all the text.
  ok = ok && text_stream.eof();
  // It should have been in range.
  ok = ok && !text_stream.fail();

  // Work around a bug in the GNU C++11 library. It will happily parse
  // "-1" for uint16_t as 65535.
  if (ok && text[0] == '-')
    ok = !ClampToZeroIfUnsignedType<T>::Clamp(value_pointer);

  return ok;
}

// Enum to indicate the parsing and encoding status.
enum class EncodeNumberStatus {
  kSuccess = 0,
  // Unsupported bit width etc.
  kUnsupported,
  // Expected type (NumberType) is not a scalar int or float, or putting a
  // negative number in an unsigned literal.
  kInvalidUsage,
  // Number value does not fit the bit width of the expected type etc.
  kInvalidText,
};

// Parses an integer value of a given |type| from the given |text| and encodes
// the number by the given |emit| function. On success, returns
// EncodeNumberStatus::kSuccess and the parsed number will be consumed by the
// given |emit| function word by word (least significant word first). On
// failure, this function returns the error code of the encoding status and
// |emit| function will not be called. If the string pointer |error_msg| is not
// a nullptr, it will be overwritten with error messages in case of failure. In
// case of success, |error_msg| will not be touched. Integers up to 64 bits are
// supported.
EncodeNumberStatus ParseAndEncodeIntegerNumber(
    const char* text, const NumberType& type,
    std::function<void(uint32_t)> emit, std::string* error_msg);

// Parses a floating point value of a given |type| from the given |text| and
// encodes the number by the given |emit| funciton. On success, returns
// EncodeNumberStatus::kSuccess and the parsed number will be consumed by the
// given |emit| function word by word (least significant word first). On
// failure, this function returns the error code of the encoding status and
// |emit| function will not be called. If the string pointer |error_msg| is not
// a nullptr, it will be overwritten with error messages in case of failure. In
// case of success, |error_msg| will not be touched. Only 16, 32 and 64 bit
// floating point numbers are supported.
EncodeNumberStatus ParseAndEncodeFloatingPointNumber(
    const char* text, const NumberType& type,
    std::function<void(uint32_t)> emit, std::string* error_msg);

// Parses an integer or floating point number of a given |type| from the given
// |text| and encodes the number by the given |emit| function. On success,
// returns EncodeNumberStatus::kSuccess and the parsed number will be consumed
// by the given |emit| function word by word (least significant word first). On
// failure, this function returns the error code of the encoding status and
// |emit| function will not be called. If the string pointer |error_msg| is not
// a nullptr, it will be overwritten with error messages in case of failure. In
// case of success, |error_msg| will not be touched. Integers up to 64 bits
// and 16/32/64 bit floating point values are supported.
EncodeNumberStatus ParseAndEncodeNumber(const char* text,
                                        const NumberType& type,
                                        std::function<void(uint32_t)> emit,
                                        std::string* error_msg);

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_PARSE_NUMBER_H_
