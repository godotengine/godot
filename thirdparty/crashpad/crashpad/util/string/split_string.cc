// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/string/split_string.h"

#include <sys/types.h>

namespace crashpad {

bool SplitStringFirst(const std::string& string,
                      char delimiter,
                      std::string* left,
                      std::string* right) {
  size_t delimiter_pos = string.find(delimiter);
  if (delimiter_pos == 0 || delimiter_pos == std::string::npos) {
    return false;
  }

  left->assign(string, 0, delimiter_pos);
  right->assign(string, delimiter_pos + 1, std::string::npos);
  return true;
}

std::vector<std::string> SplitString(const std::string& string,
                                     char delimiter) {
  std::vector<std::string> result;
  if (string.empty())
    return result;

  size_t start = 0;
  while (start != std::string::npos) {
    size_t end = string.find_first_of(delimiter, start);

    std::string part;
    if (end == std::string::npos) {
      part = string.substr(start);
      start = std::string::npos;
    } else {
      part = string.substr(start, end - start);
      start = end + 1;
    }

    result.push_back(part);
  }
  return result;
}

}  // namespace crashpad
