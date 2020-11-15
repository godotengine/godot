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

#include "util/misc/random_string.h"

#include "base/rand_util.h"

namespace crashpad {

std::string RandomString() {
  std::string random_string;
  for (int index = 0; index < 16; ++index) {
    random_string.append(1, static_cast<char>(base::RandInt('A', 'Z')));
  }
  return random_string;
}

}  // namespace crashpad
