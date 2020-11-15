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

#ifndef CRASHPAD_UTIL_MISC_REINTERPRET_BYTES_H_
#define CRASHPAD_UTIL_MISC_REINTERPRET_BYTES_H_

#include <stddef.h>

namespace crashpad {

namespace internal {

bool ReinterpretBytesImpl(const char* from,
                          size_t from_size,
                          char* to,
                          size_t to_size);

}  // namespace internal

//! \brief Copies the bytes of \a from to \a to.
//!
//! This function is similar to `bit_cast`, except that it can operate on
//! differently sized types.
//!
//! \return `true` if the copy is possible without information loss, otherwise
//!     `false` with a message logged.
template <typename From, typename To>
bool ReinterpretBytes(const From& from, To* to) {
  return internal::ReinterpretBytesImpl(reinterpret_cast<const char*>(&from),
                                        sizeof(From),
                                        reinterpret_cast<char*>(to),
                                        sizeof(To));
}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_REINTERPRET_BYTES_H_
