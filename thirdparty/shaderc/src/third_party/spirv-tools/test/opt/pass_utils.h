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

#ifndef TEST_OPT_PASS_UTILS_H_
#define TEST_OPT_PASS_UTILS_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "include/spirv-tools/libspirv.h"
#include "include/spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {

struct Message {
  spv_message_level_t level;
  const char* source_file;
  uint32_t line_number;
  uint32_t column_number;
  const char* message;
};

// Return a message consumer that can be used to check that the message produced
// are the messages in |expexted_messages|, and in the same order.
MessageConsumer GetTestMessageConsumer(std::vector<Message>& expected_messages);

// In-place substring replacement. Finds the |find_str| in the |process_str|
// and replaces the found substring with |replace_str|. Returns true if at
// least one replacement is done successfully, returns false otherwise. The
// replaced substring won't be processed again, which means: If the
// |replace_str| has |find_str| as its substring, that newly replaced part of
// |process_str| won't be processed again.
bool FindAndReplace(std::string* process_str, const std::string find_str,
                    const std::string replace_str);

// Returns true if the given string contains any debug opcode substring.
bool ContainsDebugOpcode(const char* inst);

// Returns the concatenated string from a vector of |strings|, with postfixing
// each string with the given |delimiter|. if the |skip_dictator| returns true
// for an original string, that string will be omitted.
std::string SelectiveJoin(const std::vector<const char*>& strings,
                          const std::function<bool(const char*)>& skip_dictator,
                          char delimiter = '\n');

// Concatenates a vector of strings into one string. Each string is postfixed
// with '\n'.
std::string JoinAllInsts(const std::vector<const char*>& insts);

// Concatenates a vector of strings into one string. Each string is postfixed
// with '\n'. If a string contains opcode for debug instruction, that string
// will be ignored.
std::string JoinNonDebugInsts(const std::vector<const char*>& insts);

// Returns a vector that contains the contents of |a| followed by the contents
// of |b|.
template <typename T>
std::vector<T> Concat(const std::vector<T>& a, const std::vector<T>& b) {
  std::vector<T> ret;
  std::copy(a.begin(), a.end(), back_inserter(ret));
  std::copy(b.begin(), b.end(), back_inserter(ret));
  return ret;
}

}  // namespace opt
}  // namespace spvtools

#endif  // TEST_OPT_PASS_UTILS_H_
