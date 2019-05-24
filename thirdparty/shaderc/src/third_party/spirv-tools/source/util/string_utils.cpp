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

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "source/util/string_utils.h"

namespace spvtools {
namespace utils {

std::string CardinalToOrdinal(size_t cardinal) {
  const size_t mod10 = cardinal % 10;
  const size_t mod100 = cardinal % 100;
  std::string suffix;
  if (mod10 == 1 && mod100 != 11)
    suffix = "st";
  else if (mod10 == 2 && mod100 != 12)
    suffix = "nd";
  else if (mod10 == 3 && mod100 != 13)
    suffix = "rd";
  else
    suffix = "th";

  return ToString(cardinal) + suffix;
}

std::pair<std::string, std::string> SplitFlagArgs(const std::string& flag) {
  if (flag.size() < 2) return make_pair(flag, std::string());

  // Detect the last dash before the pass name. Since we have to
  // handle single dash options (-O and -Os), count up to two dashes.
  size_t dash_ix = 0;
  if (flag[0] == '-' && flag[1] == '-')
    dash_ix = 2;
  else if (flag[0] == '-')
    dash_ix = 1;

  size_t ix = flag.find('=');
  return (ix != std::string::npos)
             ? make_pair(flag.substr(dash_ix, ix - 2), flag.substr(ix + 1))
             : make_pair(flag.substr(dash_ix), std::string());
}

}  // namespace utils
}  // namespace spvtools
