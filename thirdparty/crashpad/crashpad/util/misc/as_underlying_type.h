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

#ifndef CRASHPAD_UTIL_MISC_AS_UNDERLYING_TYPE_H_
#define CRASHPAD_UTIL_MISC_AS_UNDERLYING_TYPE_H_

#include <type_traits>

namespace crashpad {

//! \brief Casts a value to its underlying type.
//!
//! \param[in] from The value to be casted.
//! \return \a from casted to its underlying type.
template <typename From>
constexpr typename std::underlying_type<From>::type AsUnderlyingType(
    From from) {
  return static_cast<typename std::underlying_type<From>::type>(from);
}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_AS_UNDERLYING_TYPE_H_
