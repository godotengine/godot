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

#ifndef CRASHPAD_UTIL_WIN_CRITICAL_SECTION_WITH_DEBUG_INFO_H_
#define CRASHPAD_UTIL_WIN_CRITICAL_SECTION_WITH_DEBUG_INFO_H_

#include <windows.h>

namespace crashpad {

//! \brief Equivalent to `InitializeCritialSection()`, but attempts to allocate
//!     with a valid `.DebugInfo` field on versions of Windows where it's
//!     possible to do so.
//!
//! \return `true` on success, or `false` on failure with a message logged.
//!     Success means that the critical section was successfully initialized,
//!     but it does not necessarily have a valid `.DebugInfo` field.
bool InitializeCriticalSectionWithDebugInfoIfPossible(
    CRITICAL_SECTION* critical_section);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_CRITICAL_SECTION_WITH_DEBUG_INFO_H_
