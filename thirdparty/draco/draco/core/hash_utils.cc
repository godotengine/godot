// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "draco/core/hash_utils.h"

#include <cstddef>
#include <functional>
#include <limits>

namespace draco {

// Will never return 1 or 0.
uint64_t FingerprintString(const char *s, size_t len) {
  const uint64_t seed = 0x87654321;
  const int hash_loop_count = static_cast<int>(len / 8) + 1;
  uint64_t hash = seed;

  for (int i = 0; i < hash_loop_count; ++i) {
    const int off = i * 8;
    const int num_chars_left = static_cast<int>(len) - off;
    uint64_t new_hash = seed;

    if (num_chars_left > 7) {
      const int off2 = i * 8;
      new_hash = static_cast<uint64_t>(s[off2]) << 56 |
                 static_cast<uint64_t>(s[off2 + 1]) << 48 |
                 static_cast<uint64_t>(s[off2 + 2]) << 40 |
                 static_cast<uint64_t>(s[off2 + 3]) << 32 |
                 static_cast<uint64_t>(s[off2 + 4]) << 24 |
                 static_cast<uint64_t>(s[off2 + 5]) << 16 |
                 static_cast<uint64_t>(s[off2 + 6]) << 8 | s[off2 + 7];
    } else {
      for (int j = 0; j < num_chars_left; ++j) {
        new_hash |= static_cast<uint64_t>(s[off + j])
                    << (64 - ((num_chars_left - j) * 8));
      }
    }

    hash = HashCombine(new_hash, hash);
  }

  if (hash < std::numeric_limits<uint64_t>::max() - 1) {
    hash += 2;
  }
  return hash;
}
}  // namespace draco
