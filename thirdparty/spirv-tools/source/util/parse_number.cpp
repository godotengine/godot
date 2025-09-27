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

#include "source/util/parse_number.h"

#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include "source/util/hex_float.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace utils {
namespace {

// A helper class that temporarily stores error messages and dump the messages
// to a string which given as as pointer when it is destructed. If the given
// pointer is a nullptr, this class does not store error message.
class ErrorMsgStream {
 public:
  explicit ErrorMsgStream(std::string* error_msg_sink)
      : error_msg_sink_(error_msg_sink) {
    if (error_msg_sink_) stream_ = MakeUnique<std::ostringstream>();
  }
  ~ErrorMsgStream() {
    if (error_msg_sink_ && stream_) *error_msg_sink_ = stream_->str();
  }
  template <typename T>
  ErrorMsgStream& operator<<(T val) {
    if (stream_) *stream_ << val;
    return *this;
  }

 private:
  std::unique_ptr<std::ostringstream> stream_;
  // The destination string to which this class dump the error message when
  // destructor is called.
  std::string* error_msg_sink_;
};
}  // namespace

EncodeNumberStatus ParseAndEncodeIntegerNumber(
    const char* text, const NumberType& type,
    std::function<void(uint32_t)> emit, std::string* error_msg) {
  if (!text) {
    ErrorMsgStream(error_msg) << "The given text is a nullptr";
    return EncodeNumberStatus::kInvalidText;
  }

  if (!IsIntegral(type)) {
    ErrorMsgStream(error_msg) << "The expected type is not a integer type";
    return EncodeNumberStatus::kInvalidUsage;
  }

  const uint32_t bit_width = AssumedBitWidth(type);

  if (bit_width > 64) {
    ErrorMsgStream(error_msg)
        << "Unsupported " << bit_width << "-bit integer literals";
    return EncodeNumberStatus::kUnsupported;
  }

  // Either we are expecting anything or integer.
  bool is_negative = text[0] == '-';
  bool can_be_signed = IsSigned(type);

  if (is_negative && !can_be_signed) {
    ErrorMsgStream(error_msg)
        << "Cannot put a negative number in an unsigned literal";
    return EncodeNumberStatus::kInvalidUsage;
  }

  const bool is_hex = text[0] == '0' && (text[1] == 'x' || text[1] == 'X');

  uint64_t decoded_bits;
  if (is_negative) {
    int64_t decoded_signed = 0;

    if (!ParseNumber(text, &decoded_signed)) {
      ErrorMsgStream(error_msg) << "Invalid signed integer literal: " << text;
      return EncodeNumberStatus::kInvalidText;
    }

    if (!CheckRangeAndIfHexThenSignExtend(decoded_signed, type, is_hex,
                                          &decoded_signed)) {
      ErrorMsgStream(error_msg)
          << "Integer " << (is_hex ? std::hex : std::dec) << std::showbase
          << decoded_signed << " does not fit in a " << std::dec << bit_width
          << "-bit " << (IsSigned(type) ? "signed" : "unsigned") << " integer";
      return EncodeNumberStatus::kInvalidText;
    }
    decoded_bits = decoded_signed;
  } else {
    // There's no leading minus sign, so parse it as an unsigned integer.
    if (!ParseNumber(text, &decoded_bits)) {
      ErrorMsgStream(error_msg) << "Invalid unsigned integer literal: " << text;
      return EncodeNumberStatus::kInvalidText;
    }
    if (!CheckRangeAndIfHexThenSignExtend(decoded_bits, type, is_hex,
                                          &decoded_bits)) {
      ErrorMsgStream(error_msg)
          << "Integer " << (is_hex ? std::hex : std::dec) << std::showbase
          << decoded_bits << " does not fit in a " << std::dec << bit_width
          << "-bit " << (IsSigned(type) ? "signed" : "unsigned") << " integer";
      return EncodeNumberStatus::kInvalidText;
    }
  }
  if (bit_width > 32) {
    uint32_t low = uint32_t(0x00000000ffffffff & decoded_bits);
    uint32_t high = uint32_t((0xffffffff00000000 & decoded_bits) >> 32);
    emit(low);
    emit(high);
  } else {
    emit(uint32_t(decoded_bits));
  }
  return EncodeNumberStatus::kSuccess;
}

EncodeNumberStatus ParseAndEncodeFloatingPointNumber(
    const char* text, const NumberType& type,
    std::function<void(uint32_t)> emit, std::string* error_msg) {
  if (!text) {
    ErrorMsgStream(error_msg) << "The given text is a nullptr";
    return EncodeNumberStatus::kInvalidText;
  }

  if (!IsFloating(type)) {
    ErrorMsgStream(error_msg) << "The expected type is not a float type";
    return EncodeNumberStatus::kInvalidUsage;
  }

  const auto bit_width = AssumedBitWidth(type);
  switch (bit_width) {
    case 16: {
      HexFloat<FloatProxy<Float16>> hVal(0);
      if (!ParseNumber(text, &hVal)) {
        ErrorMsgStream(error_msg) << "Invalid 16-bit float literal: " << text;
        return EncodeNumberStatus::kInvalidText;
      }
      // getAsFloat will return the Float16 value, and get_value
      // will return a uint16_t representing the bits of the float.
      // The encoding is therefore correct from the perspective of the SPIR-V
      // spec since the top 16 bits will be 0.
      emit(static_cast<uint32_t>(hVal.value().getAsFloat().get_value()));
      return EncodeNumberStatus::kSuccess;
    } break;
    case 32: {
      HexFloat<FloatProxy<float>> fVal(0.0f);
      if (!ParseNumber(text, &fVal)) {
        ErrorMsgStream(error_msg) << "Invalid 32-bit float literal: " << text;
        return EncodeNumberStatus::kInvalidText;
      }
      emit(BitwiseCast<uint32_t>(fVal));
      return EncodeNumberStatus::kSuccess;
    } break;
    case 64: {
      HexFloat<FloatProxy<double>> dVal(0.0);
      if (!ParseNumber(text, &dVal)) {
        ErrorMsgStream(error_msg) << "Invalid 64-bit float literal: " << text;
        return EncodeNumberStatus::kInvalidText;
      }
      uint64_t decoded_val = BitwiseCast<uint64_t>(dVal);
      uint32_t low = uint32_t(0x00000000ffffffff & decoded_val);
      uint32_t high = uint32_t((0xffffffff00000000 & decoded_val) >> 32);
      emit(low);
      emit(high);
      return EncodeNumberStatus::kSuccess;
    } break;
    default:
      break;
  }
  ErrorMsgStream(error_msg)
      << "Unsupported " << bit_width << "-bit float literals";
  return EncodeNumberStatus::kUnsupported;
}

EncodeNumberStatus ParseAndEncodeNumber(const char* text,
                                        const NumberType& type,
                                        std::function<void(uint32_t)> emit,
                                        std::string* error_msg) {
  if (!text) {
    ErrorMsgStream(error_msg) << "The given text is a nullptr";
    return EncodeNumberStatus::kInvalidText;
  }

  if (IsUnknown(type)) {
    ErrorMsgStream(error_msg)
        << "The expected type is not a integer or float type";
    return EncodeNumberStatus::kInvalidUsage;
  }

  // If we explicitly expect a floating-point number, we should handle that
  // first.
  if (IsFloating(type)) {
    return ParseAndEncodeFloatingPointNumber(text, type, emit, error_msg);
  }

  return ParseAndEncodeIntegerNumber(text, type, emit, error_msg);
}

}  // namespace utils
}  // namespace spvtools
