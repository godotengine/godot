// Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef SOURCE_UTIL_HASH_COMBINE_H_
#define SOURCE_UTIL_HASH_COMBINE_H_

#include <cstddef>
#include <functional>
#include <vector>

namespace spvtools {
namespace utils {

// Helpers for incrementally computing hashes.
// For reference, see
// http://open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf

template <typename T>
inline size_t hash_combine(std::size_t seed, const T& val) {
  return seed ^ (std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename T>
inline size_t hash_combine(std::size_t hash, const std::vector<T>& vals) {
  for (const T& val : vals) {
    hash = hash_combine(hash, val);
  }
  return hash;
}

inline size_t hash_combine(std::size_t hash) { return hash; }

template <typename T, typename... Types>
inline size_t hash_combine(std::size_t hash, const T& val,
                           const Types&... args) {
  return hash_combine(hash_combine(hash, val), args...);
}

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_HASH_COMBINE_H_
