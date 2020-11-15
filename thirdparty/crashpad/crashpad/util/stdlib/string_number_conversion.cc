// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/stdlib/string_number_conversion.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include <limits>

#include "base/logging.h"

namespace {

template <typename TIntType, typename TLongType>
struct StringToIntegerTraits {
  using IntType = TIntType;
  using LongType = TLongType;
  static void TypeCheck() {
    static_assert(std::numeric_limits<TIntType>::is_integer &&
                      std::numeric_limits<TLongType>::is_integer,
                  "IntType and LongType must be integer");
    static_assert(std::numeric_limits<TIntType>::is_signed ==
                      std::numeric_limits<TLongType>::is_signed,
                  "IntType and LongType signedness must agree");
    static_assert(std::numeric_limits<TIntType>::min() >=
                          std::numeric_limits<TLongType>::min() &&
                      std::numeric_limits<TIntType>::min() <
                          std::numeric_limits<TLongType>::max(),
                  "IntType min must be in LongType range");
    static_assert(std::numeric_limits<TIntType>::max() >
                          std::numeric_limits<TLongType>::min() &&
                      std::numeric_limits<TIntType>::max() <=
                          std::numeric_limits<TLongType>::max(),
                  "IntType max must be in LongType range");
  }
};

template <typename TIntType, typename TLongType>
struct StringToSignedIntegerTraits
    : public StringToIntegerTraits<TIntType, TLongType> {
  static void TypeCheck() {
    static_assert(std::numeric_limits<TIntType>::is_signed,
                  "StringToSignedTraits IntType must be signed");
    return super::TypeCheck();
  }
  static bool IsNegativeOverflow(TLongType value) {
    return value < std::numeric_limits<TIntType>::min();
  }

 private:
  using super = StringToIntegerTraits<TIntType, TLongType>;
};

template <typename TIntType, typename TLongType>
struct StringToUnsignedIntegerTraits
    : public StringToIntegerTraits<TIntType, TLongType> {
  static void TypeCheck() {
    static_assert(!std::numeric_limits<TIntType>::is_signed,
                  "StringToUnsignedTraits IntType must be unsigned");
    return super::TypeCheck();
  }
  static bool IsNegativeOverflow(TLongType value) { return false; }

 private:
  using super = StringToIntegerTraits<TIntType, TLongType>;
};

struct StringToIntTraits : public StringToSignedIntegerTraits<int, long> {
  static LongType Convert(const char* str, char** end, int base) {
    return strtol(str, end, base);
  }
};

struct StringToUnsignedIntTraits
    : public StringToUnsignedIntegerTraits<unsigned int, unsigned long> {
  static LongType Convert(const char* str, char** end, int base) {
    if (str[0] == '-') {
      *end = const_cast<char*>(str);
      return 0;
    }
    return strtoul(str, end, base);
  }
};

struct StringToInt64Traits
    : public StringToSignedIntegerTraits<int64_t, int64_t> {
  static LongType Convert(const char* str, char** end, int base) {
    return strtoll(str, end, base);
  }
};

struct StringToUnsignedInt64Traits
    : public StringToUnsignedIntegerTraits<uint64_t, uint64_t> {
  static LongType Convert(const char* str, char** end, int base) {
    if (str[0] == '-') {
      *end = const_cast<char*>(str);
      return 0;
    }
    return strtoull(str, end, base);
  }
};

template <typename Traits>
bool StringToIntegerInternal(const std::string& string,
                             typename Traits::IntType* number) {
  using IntType = typename Traits::IntType;
  using LongType = typename Traits::LongType;

  Traits::TypeCheck();

  if (string.empty() || isspace(string[0])) {
    return false;
  }

  errno = 0;
  char* end;
  LongType result = Traits::Convert(string.data(), &end, 0);
  if (Traits::IsNegativeOverflow(result) ||
      result > std::numeric_limits<IntType>::max() ||
      errno == ERANGE ||
      end != string.data() + string.length()) {
    return false;
  }
  *number = result;
  return true;
}

}  // namespace

namespace crashpad {

bool StringToNumber(const std::string& string, int* number) {
  return StringToIntegerInternal<StringToIntTraits>(string, number);
}

bool StringToNumber(const std::string& string, unsigned int* number) {
  return StringToIntegerInternal<StringToUnsignedIntTraits>(string, number);
}

bool StringToNumber(const std::string& string, int64_t* number) {
  return StringToIntegerInternal<StringToInt64Traits>(string, number);
}

bool StringToNumber(const std::string& string, uint64_t* number) {
  return StringToIntegerInternal<StringToUnsignedInt64Traits>(string, number);
}

}  // namespace crashpad
