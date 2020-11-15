// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/misc/lexing.h"

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <limits>

#include "base/strings/string_number_conversions.h"
#include "base/strings/string_piece.h"

namespace crashpad {

namespace {

#define MAKE_ADAPTER(type, function)                                        \
  bool ConvertStringToNumber(const base::StringPiece& input, type* value) { \
    return function(input, value);                                          \
  }
MAKE_ADAPTER(int, base::StringToInt);
MAKE_ADAPTER(unsigned int, base::StringToUint);
MAKE_ADAPTER(int64_t, base::StringToInt64);
MAKE_ADAPTER(uint64_t, base::StringToUint64);
#undef MAKE_ADAPTER

}  // namespace

bool AdvancePastPrefix(const char** input, const char* pattern) {
  size_t length = strlen(pattern);
  if (strncmp(*input, pattern, length) == 0) {
    *input += length;
    return true;
  }
  return false;
}

template <typename T>
bool AdvancePastNumber(const char** input, T* value) {
  size_t length = 0;
  if (std::numeric_limits<T>::is_signed && **input == '-') {
    ++length;
  }
  while (isdigit((*input)[length])) {
    ++length;
  }
  bool success =
      ConvertStringToNumber(base::StringPiece(*input, length), value);
  if (success) {
    *input += length;
    return true;
  }
  return false;
}

template bool AdvancePastNumber(const char** input, int* value);
template bool AdvancePastNumber(const char** input, unsigned int* value);
template bool AdvancePastNumber(const char** input, int64_t* value);
template bool AdvancePastNumber(const char** input, uint64_t* value);

}  // namespace crashpad
