// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_UTIL_STRING_UTILS_H_
#define SOURCE_UTIL_STRING_UTILS_H_

#include <sstream>
#include <string>

#include "source/util/string_utils.h"

namespace spvtools {
namespace utils {

// Converts arithmetic value |val| to its default string representation.
template <class T>
std::string ToString(T val) {
  static_assert(
      std::is_arithmetic<T>::value,
      "spvtools::utils::ToString is restricted to only arithmetic values");
  std::stringstream os;
  os << val;
  return os.str();
}

// Converts cardinal number to ordinal number string.
std::string CardinalToOrdinal(size_t cardinal);

// Splits the string |flag|, of the form '--pass_name[=pass_args]' into two
// strings "pass_name" and "pass_args".  If |flag| has no arguments, the second
// string will be empty.
std::pair<std::string, std::string> SplitFlagArgs(const std::string& flag);

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_STRING_UTILS_H_
