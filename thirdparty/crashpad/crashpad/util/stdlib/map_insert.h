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

#ifndef CRASHPAD_UTIL_STDLIB_MAP_INSERT_H_
#define CRASHPAD_UTIL_STDLIB_MAP_INSERT_H_

#include <map>
#include <utility>

namespace crashpad {

//! \brief Inserts a mapping from \a key to \a value into \a map, or replaces
//!     an existing mapping so that \a key maps to \a value.
//!
//! This behaves similarly to `std::map<>::%insert_or_assign()` proposed for
//! C++17, except that the \a old_value parameter is added.
//!
//! \param[in,out] map The map to operate on.
//! \param[in] key The key that should be mapped to \a value.
//! \param[in] value The value that \a key should map to.
//! \param[out] old_value If \a key was previously present in \a map, this will
//!     be set to its previous value. This parameter is optional and may be
//!     `nullptr` if this information is not required.
//!
//! \return `false` if \a key was previously present in \a map. If \a old_value
//!     is not `nullptr`, it will be set to the previous value. `true` if \a
//!     key was not present in the map and was inserted.
template <typename T>
bool MapInsertOrReplace(T* map,
                        const typename T::key_type& key,
                        const typename T::mapped_type& value,
                        typename T::mapped_type* old_value) {
  const auto result = map->insert(std::make_pair(key, value));
  if (!result.second) {
    if (old_value) {
      *old_value = result.first->second;
    }
    result.first->second = value;
  }
  return result.second;
}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STDLIB_MAP_INSERT_H_
