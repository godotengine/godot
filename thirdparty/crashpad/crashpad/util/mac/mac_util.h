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

#ifndef CRASHPAD_UTIL_MAC_MAC_UTIL_H_
#define CRASHPAD_UTIL_MAC_MAC_UTIL_H_

#include <string>

namespace crashpad {

//! \brief Returns the version of the running operating system.
//!
//! \return The minor version of the operating system, such as `12` for macOS
//!     10.12.1.
//!
//! \note This is similar to the base::mac::IsOS*() family of functions, but
//!     is provided for situations where the caller needs to obtain version
//!     information beyond what is provided by Chromium’s base, or for when the
//!     caller needs the actual minor version value.
int MacOSXMinorVersion();

//! \brief Returns the version of the running operating system.
//!
//! All parameters are required. No parameter may be `nullptr`.
//!
//! \param[out] major The major version of the operating system, such as `10`
//!     for macOS 10.12.1.
//! \param[out] minor The major version of the operating system, such as `12`
//!     for macOS 10.12.1.
//! \param[out] bugfix The bugfix version of the operating system, such as `1`
//!     for macOS 10.12.1.
//! \param[out] build The operating system’s build string, such as `"16B2657"`
//!     for macOS 10.12.1.
//! \param[out] server `true` for a macOS Server installation, `false` otherwise
//!     (for a desktop/laptop, client, or workstation system).
//! \param[out] version_string A string representing the full operating system
//!     version, such as `"macOS 10.12.1 (16B2657)"`.
//!
//! \return `true` on success, `false` on failure, with an error message logged.
//!     A failure is considered to have occurred if any element could not be
//!     determined. When this happens, their values will be untouched, but other
//!     values that could be determined will still be set properly.
bool MacOSXVersion(int* major,
                   int* minor,
                   int* bugfix,
                   std::string* build,
                   bool* server,
                   std::string* version_string);

//! \brief Returns the model name and board ID of the running system.
//!
//! \param[out] model The system’s model name. A mid-2012 15" MacBook Pro would
//!     report “MacBookPro10,1”.
//! \param[out] board_id The system’s board ID. A mid-2012 15" MacBook Pro would
//!     report “Mac-C3EC7CD22292981F”.
//!
//! If a value cannot be determined, its string is cleared.
void MacModelAndBoard(std::string* model, std::string* board_id);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MAC_MAC_UTIL_H_
