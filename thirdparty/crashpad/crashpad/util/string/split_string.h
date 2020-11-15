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

#ifndef CRASHPAD_UTIL_STRING_SPLIT_STRING_H_
#define CRASHPAD_UTIL_STRING_SPLIT_STRING_H_

#include <string>
#include <vector>

namespace crashpad {

//! \brief Splits a string into two parts at the first delimiter found.
//!
//! \param[in] string The string to split.
//! \param[in] delimiter The delimiter to split at.
//! \param[out] left The portion of \a string up to, but not including, the
//!     first \a delimiter character.
//! \param[out] right The portion of \a string after the first \a delimiter
//!     character.
//!
//! \return `true` if \a string was split successfully. `false` if \a string
//!     did not contain a \a delimiter character or began with a \a delimiter
//!     character.
bool SplitStringFirst(const std::string& string,
                      char delimiter,
                      std::string* left,
                      std::string* right);

//! \brief Splits a string into multiple parts on the given delimiter.
//!
//! \param[in] string The string to split.
//! \param[in] delimiter The delimiter to split at.
//!
//! \return The individual parts of the string.
std::vector<std::string> SplitString(const std::string& string, char delimiter);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STRING_SPLIT_STRING_H_
