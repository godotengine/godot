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

#include "test/opt/pass_utils.h"

#include <algorithm>
#include <sstream>

namespace spvtools {
namespace opt {
namespace {

// Well, this is another place requiring the knowledge of the grammar and can be
// stale when SPIR-V is updated. It would be nice to automatically generate
// this, but the cost is just too high.

const char* kDebugOpcodes[] = {
    // clang-format off
    "OpSourceContinued", "OpSource", "OpSourceExtension",
    "OpName", "OpMemberName", "OpString",
    "OpLine", "OpNoLine", "OpModuleProcessed"
    // clang-format on
};

}  // anonymous namespace

MessageConsumer GetTestMessageConsumer(
    std::vector<Message>& expected_messages) {
  return [&expected_messages](spv_message_level_t level, const char* source,
                              const spv_position_t& position,
                              const char* message) {
    EXPECT_TRUE(!expected_messages.empty());
    if (expected_messages.empty()) {
      return;
    }

    EXPECT_EQ(expected_messages[0].level, level);
    EXPECT_EQ(expected_messages[0].line_number, position.line);
    EXPECT_EQ(expected_messages[0].column_number, position.column);
    EXPECT_STREQ(expected_messages[0].source_file, source);
    EXPECT_STREQ(expected_messages[0].message, message);

    expected_messages.erase(expected_messages.begin());
  };
}

bool FindAndReplace(std::string* process_str, const std::string find_str,
                    const std::string replace_str) {
  if (process_str->empty() || find_str.empty()) {
    return false;
  }
  bool replaced = false;
  // Note this algorithm has quadratic time complexity. It is OK for test cases
  // with short strings, but might not fit in other contexts.
  for (size_t pos = process_str->find(find_str, 0); pos != std::string::npos;
       pos = process_str->find(find_str, pos)) {
    process_str->replace(pos, find_str.length(), replace_str);
    pos += replace_str.length();
    replaced = true;
  }
  return replaced;
}

bool ContainsDebugOpcode(const char* inst) {
  return std::any_of(std::begin(kDebugOpcodes), std::end(kDebugOpcodes),
                     [inst](const char* op) {
                       return std::string(inst).find(op) != std::string::npos;
                     });
}

std::string SelectiveJoin(const std::vector<const char*>& strings,
                          const std::function<bool(const char*)>& skip_dictator,
                          char delimiter) {
  std::ostringstream oss;
  for (const auto* str : strings) {
    if (!skip_dictator(str)) oss << str << delimiter;
  }
  return oss.str();
}

std::string JoinAllInsts(const std::vector<const char*>& insts) {
  return SelectiveJoin(insts, [](const char*) { return false; });
}

std::string JoinNonDebugInsts(const std::vector<const char*>& insts) {
  return SelectiveJoin(
      insts, [](const char* inst) { return ContainsDebugOpcode(inst); });
}

}  // namespace opt
}  // namespace spvtools
