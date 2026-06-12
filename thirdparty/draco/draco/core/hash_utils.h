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
#ifndef DRACO_CORE_HASH_UTILS_H_
#define DRACO_CORE_HASH_UTILS_H_

#include <stdint.h>

#include <cstddef>
#include <functional>

namespace draco {

template <typename T1, typename T2>
size_t HashCombine(T1 a, T2 b) {
  const size_t hash1 = std::hash<T1>()(a);
  const size_t hash2 = std::hash<T2>()(b);
  return (hash1 << 2) ^ (hash2 << 1);
}

template <typename T>
size_t HashCombine(T a, size_t hash) {
  const size_t hasha = std::hash<T>()(a);
  return (hash) ^ (hasha + 239);
}

inline uint64_t HashCombine(uint64_t a, uint64_t b) {
  return (a + 1013) ^ (b + 107) << 1;
}

// Will never return 1 or 0.
uint64_t FingerprintString(const char *s, size_t len);

// Hash for std::array.
template <typename T>
struct HashArray {
  size_t operator()(const T &a) const {
    size_t hash = 79;  // Magic number.
    for (unsigned int i = 0; i < std::tuple_size<T>::value; ++i) {
      hash = HashCombine(hash, ValueHash(a[i]));
    }
    return hash;
  }

  template <typename V>
  size_t ValueHash(const V &val) const {
    return std::hash<V>()(val);
  }
};

}  // namespace draco

#endif  // DRACO_CORE_HASH_UTILS_H_
