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

#include <assert.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

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

// Encodes a string as a sequence of words, using the SPIR-V encoding, appending
// to an existing vector.
inline void AppendToVector(const std::string& input,
                           std::vector<uint32_t>* result) {
  uint32_t word = 0;
  size_t num_bytes = input.size();
  // SPIR-V strings are null-terminated.  The byte_index == num_bytes
  // case is used to push the terminating null byte.
  for (size_t byte_index = 0; byte_index <= num_bytes; byte_index++) {
    const auto new_byte =
        (byte_index < num_bytes ? uint8_t(input[byte_index]) : uint8_t(0));
    word |= (new_byte << (8 * (byte_index % sizeof(uint32_t))));
    if (3 == (byte_index % sizeof(uint32_t))) {
      result->push_back(word);
      word = 0;
    }
  }
  // Emit a trailing partial word.
  if ((num_bytes + 1) % sizeof(uint32_t)) {
    result->push_back(word);
  }
}

// Encodes a string as a sequence of words, using the SPIR-V encoding.
inline std::vector<uint32_t> MakeVector(const std::string& input) {
  std::vector<uint32_t> result;
  AppendToVector(input, &result);
  return result;
}

// Decode a string from a sequence of words between first and last, using the
// SPIR-V encoding. Assert that a terminating 0-byte was found (unless
// assert_found_terminating_null is passed as false).
template <class InputIt>
inline std::string MakeString(InputIt first, InputIt last,
                              bool assert_found_terminating_null = true) {
  std::string result;
  constexpr size_t kCharsPerWord = sizeof(*first);
  static_assert(kCharsPerWord == 4, "expect 4-byte word");

  for (InputIt pos = first; pos != last; ++pos) {
    uint32_t word = *pos;
    for (size_t byte_index = 0; byte_index < kCharsPerWord; byte_index++) {
      uint32_t extracted_word = (word >> (8 * byte_index)) & 0xFF;
      char c = static_cast<char>(extracted_word);
      if (c == 0) {
        return result;
      }
      result += c;
    }
  }
  assert(!assert_found_terminating_null &&
         "Did not find terminating null for the string.");
  (void)assert_found_terminating_null; /* No unused parameters in release
                                          builds. */
  return result;
}

// Decode a string from a sequence of words in a vector, using the SPIR-V
// encoding.
template <class VectorType>
inline std::string MakeString(const VectorType& words,
                              bool assert_found_terminating_null = true) {
  return MakeString(words.cbegin(), words.cend(),
                    assert_found_terminating_null);
}

// Decode a string from array words, consuming up to count words, using the
// SPIR-V encoding.
inline std::string MakeString(const uint32_t* words, size_t num_words,
                              bool assert_found_terminating_null = true) {
  return MakeString(words, words + num_words, assert_found_terminating_null);
}

// Check if str starts with prefix (only included since C++20)
inline bool starts_with(const std::string& str, const char* prefix) {
  return 0 == str.compare(0, std::strlen(prefix), prefix);
}

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_STRING_UTILS_H_
