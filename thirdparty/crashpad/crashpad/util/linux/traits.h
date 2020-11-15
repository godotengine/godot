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

#ifndef CRASHPAD_UTIL_LINUX_TRAITS_H_
#define CRASHPAD_UTIL_LINUX_TRAITS_H_

#include <stdint.h>

namespace crashpad {

struct Traits32 {
  using Nothing = char[0];
  using Address = uint32_t;
  using Long = int32_t;
  using ULong = uint32_t;
  using Clock = Long;
  using Size = uint32_t;
  using Char_64Only = Nothing;
  using ULong_32Only = ULong;
  using ULong_64Only = Nothing;
  using UInteger32_64Only = Nothing;
};

struct Traits64 {
  using Nothing = char[0];
  using Address = uint64_t;
  using Long = int64_t;
  using ULong = uint64_t;
  using Clock = Long;
  using Size = uint64_t;
  using Char_64Only = char;
  using ULong_32Only = Nothing;
  using ULong_64Only = ULong;
  using UInteger32_64Only = uint32_t;
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_TRAITS_H_
