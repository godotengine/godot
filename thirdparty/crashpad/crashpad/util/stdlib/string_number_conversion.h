// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_STDLIB_STRING_NUMBER_CONVERSION_H_
#define CRASHPAD_UTIL_STDLIB_STRING_NUMBER_CONVERSION_H_

#include <string>

namespace crashpad {

// Convert between strings and numbers.
//
// These functions will only set *number if a perfect conversion can be
// performed. A perfect conversion contains no leading or trailing characters
// (including whitespace) other than the number to convert, and does not
// overflow the targeted data type. If a perfect conversion is possible, *number
// is set and these functions return true. Otherwise, they return false.
//
// The interface in base/strings/string_number_conversions.h doesn’t allow
// arbitrary bases based on whether the string begins with prefixes such as "0x"
// as strtol does with base = 0. The functions here are implemented on the
// strtol family with base = 0, and thus do accept such input.

//! \{
//! \brief Convert a string to a number.
//!
//! A conversion will only be performed if it can be done perfectly: if \a
//! string contains no leading or trailing characters (including whitespace)
//! other than the number to convert, and does not overflow the targeted data
//! type.
//!
//! \param[in] string The string to convert to a number. As in `strtol()` with a
//!     `base` parameter of `0`, the string is treated as decimal unless it
//!     begins with a `"0x"` or `"0X"` prefix, in which case it is treated as
//!     hexadecimal, or a `"0"` prefix, in which case it is treated as octal.
//! \param[out] number The converted number. This will only be set if a perfect
//!     conversion can be performed.
//!
//! \return `true` if a perfect conversion could be performed, with \a number
//!     set appropriately. `false` if a perfect conversion was not possible.
//!
//! \note The interface in `base/strings/string_number_conversions.h` doesn’t
//!     allow arbitrary bases based on whether the string begins with a prefix
//!     indicating its base. The functions here are provided for situations
//!     where such prefix recognition is desirable.
bool StringToNumber(const std::string& string, int* number);
bool StringToNumber(const std::string& string, unsigned int* number);
bool StringToNumber(const std::string& string, int64_t* number);
bool StringToNumber(const std::string& string, uint64_t* number);
//! \}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STDLIB_STRING_NUMBER_CONVERSION_H_
