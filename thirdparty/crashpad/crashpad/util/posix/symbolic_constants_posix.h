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

#ifndef CRASHPAD_UTIL_POSIX_SYMBOLIC_CONSTANTS_POSIX_H_
#define CRASHPAD_UTIL_POSIX_SYMBOLIC_CONSTANTS_POSIX_H_

#include <string>

#include "base/strings/string_piece.h"
#include "util/misc/symbolic_constants_common.h"

namespace crashpad {

//! \brief Converts a POSIX signal value to a textual representation.
//!
//! \param[in] signal The signal value to convert.
//! \param[in] options Options affecting the conversion. ::kUseOr is ignored.
//!     For ::kUnknownIsNumeric, the format is `"%d"`.
//!
//! \return The converted string.
std::string SignalToString(int signal, SymbolicConstantToStringOptions options);

//! \brief Converts a string to its corresponding POSIX signal value.
//!
//! \param[in] string The string to convert.
//! \param[in] options Options affecting the conversion. ::kAllowOr is ignored.
//! \param[out] signal The converted POSIX signal value.
//!
//! \return `true` on success, `false` if \a string could not be converted as
//!     requested.
bool StringToSignal(const base::StringPiece& string,
                    StringToSymbolicConstantOptions options,
                    int* signal);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_SYMBOLIC_CONSTANTS_POSIX_H_
