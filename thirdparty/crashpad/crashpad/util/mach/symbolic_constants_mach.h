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

#ifndef CRASHPAD_UTIL_MACH_SYMBOLIC_CONSTANTS_MACH_H_
#define CRASHPAD_UTIL_MACH_SYMBOLIC_CONSTANTS_MACH_H_

#include <mach/mach.h>

#include <string>

#include "base/strings/string_piece.h"
#include "util/misc/symbolic_constants_common.h"

namespace crashpad {

//! \brief Converts a Mach exception value to a textual representation.
//!
//! \param[in] exception The Mach exception value to convert.
//! \param[in] options Options affecting the conversion. ::kUseOr is ignored.
//!     For ::kUnknownIsNumeric, the format is `"%d"`.
//!
//! \return The converted string.
std::string ExceptionToString(exception_type_t exception,
                              SymbolicConstantToStringOptions options);

//! \brief Converts a string to its corresponding Mach exception value.
//!
//! \param[in] string The string to convert.
//! \param[in] options Options affecting the conversion. ::kAllowOr is ignored.
//! \param[out] exception The converted Mach exception value.
//!
//! \return `true` on success, `false` if \a string could not be converted as
//!     requested.
bool StringToException(const base::StringPiece& string,
                       StringToSymbolicConstantOptions options,
                       exception_type_t* exception);

//! \brief Converts a Mach exception mask value to a textual representation.
//!
//! \param[in] exception_mask The Mach exception mask value to convert.
//! \param[in] options Options affecting the conversion. ::kUseOr is honored.
//!     For ::kUnknownIsNumeric, the format is `"%#x"`.
//!
//! \return The converted string.
std::string ExceptionMaskToString(exception_mask_t exception_mask,
                                  SymbolicConstantToStringOptions options);

//! \brief Converts a string to its corresponding Mach exception mask value.
//!
//! \param[in] string The string to convert.
//! \param[in] options Options affecting the conversion. ::kAllowOr is honored.
//! \param[out] exception_mask The converted Mach exception mask value.
//!
//! \return `true` on success, `false` if \a string could not be converted as
//!     requested.
bool StringToExceptionMask(const base::StringPiece& string,
                           StringToSymbolicConstantOptions options,
                           exception_mask_t* exception_mask);

//! \brief Converts a Mach exception behavior value to a textual representation.
//!
//! \param[in] behavior The Mach exception behavior value to convert.
//! \param[in] options Options affecting the conversion. ::kUseOr is ignored.
//!     `MACH_EXCEPTION_CODES` can always be ORed in, but no other values can be
//!     ORed with each other. For ::kUnknownIsNumeric, the format is `"%#x"`.
//!
//! \return The converted string.
std::string ExceptionBehaviorToString(exception_behavior_t behavior,
                                      SymbolicConstantToStringOptions options);

//! \brief Converts a string to its corresponding Mach exception behavior value.
//!
//! \param[in] string The string to convert.
//! \param[in] options Options affecting the conversion. ::kAllowOr is ignored.
//!     `MACH_EXCEPTION_CODES` can always be ORed in, but no other values can be
//!     ORed with each other.
//! \param[out] behavior The converted Mach exception behavior value.
//!
//! \return `true` on success, `false` if \a string could not be converted as
//!     requested.
bool StringToExceptionBehavior(const base::StringPiece& string,
                               StringToSymbolicConstantOptions options,
                               exception_behavior_t* behavior);

//! \brief Converts a thread state flavor value to a textual representation.
//!
//! \param[in] flavor The thread state flavor value to convert.
//! \param[in] options Options affecting the conversion. ::kUseOr is ignored.
//!     For ::kUnknownIsNumeric, the format is `"%d"`.
//!
//! \return The converted string.
std::string ThreadStateFlavorToString(thread_state_flavor_t flavor,
                                      SymbolicConstantToStringOptions options);

//! \brief Converts a string to its corresponding thread state flavor value.
//!
//! \param[in] string The string to convert.
//! \param[in] options Options affecting the conversion. ::kAllowOr is ignored.
//! \param[out] flavor The converted thread state flavor value.
//!
//! \return `true` on success, `false` if \a string could not be converted as
//!     requested.
bool StringToThreadStateFlavor(const base::StringPiece& string,
                               StringToSymbolicConstantOptions options,
                               thread_state_flavor_t* flavor);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_SYMBOLIC_CONSTANTS_MACH_H_
