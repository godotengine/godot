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

#ifndef CRASHPAD_UTIL_MISC_TRI_STATE_H_
#define CRASHPAD_UTIL_MISC_TRI_STATE_H_

#include <stdint.h>

namespace crashpad {

//! \brief A tri-state value that can be unset, on, or off.
enum class TriState : uint8_t {
  //! \brief The value has not explicitly been set.
  //!
  //! To allow a zero-initialized value to have this behavior, this must have
  //! the value `0`.
  kUnset = 0,

  //! \brief The value has explicitly been set to on, or a behavior has
  //!     explicitly been enabled.
  kEnabled,

  //! \brief The value has explicitly been set to off, or a behavior has
  //!     explicitly been disabled.
  kDisabled,
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_TRI_STATE_H_
