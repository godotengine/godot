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

#ifndef CRASHPAD_UTIL_MISC_LEXING_H_
#define CRASHPAD_UTIL_MISC_LEXING_H_

namespace crashpad {

//! \brief Match a pattern at the start of a char string.
//!
//! \param[in,out] input A pointer to the char string to match against. \a input
//!     is advanced past the matched pattern if it is found.
//! \param[in] pattern The pattern to match at the start of \a input.
//! \return `true` if the pattern is matched exactly and \a input is advanced,
//!     otherwise `false`.
bool AdvancePastPrefix(const char** input, const char* pattern);

//! \brief Convert a prefix of a char string to a numeric value.
//!
//! Valid values are positive or negative decimal numbers, matching the regular
//! expression "-?\d+", and within the limits of T.
//!
//! \param[in,out] input A pointer to the char string to match against. \a input
//!     is advanced past the number if one is found.
//! \param[out] value The converted number, if one is found.
//! \return `true` if a number is found at the start of \a input and \a input is
//!     advanced, otherwise `false`.
template <typename T>
bool AdvancePastNumber(const char** input, T* value);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_LEXING_H
