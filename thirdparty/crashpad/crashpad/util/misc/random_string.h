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

#ifndef CRASHPAD_UTIL_MISC_RANDOM_STRING_H_
#define CRASHPAD_UTIL_MISC_RANDOM_STRING_H_

#include <string>

namespace crashpad {

//! \brief Returns a random string.
//!
//! The string consists of 16 uppercase characters chosen at random. The
//! returned string has over 75 bits of randomness (26<sup>16</sup> &gt;
//! 2<sup>75</sup>).
std::string RandomString();

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_RANDOM_STRING_H_
