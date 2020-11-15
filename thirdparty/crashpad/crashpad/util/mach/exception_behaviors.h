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

#ifndef CRASHPAD_UTIL_MACH_EXCEPTION_BEHAVIORS_H_
#define CRASHPAD_UTIL_MACH_EXCEPTION_BEHAVIORS_H_

#include <mach/mach.h>

namespace crashpad {

//! \brief Determines whether \a behavior indicates an exception behavior that
//!     carries thread state information.
//!
//! When this function returns `true`, an exception message of \a behavior will
//! carry thread state information. Its \a flavor, \a old_state, \a
//! old_state_count, \a new_state, and \a new_state_count fields will be valid.
//! When this function returns `false`, these fields will not be valid.
//!
//! Exception behaviors that carry thread state information are
//! `EXCEPTION_STATE` and `EXCEPTION_STATE_IDENTITY`. `MACH_EXCEPTION_CODES` may
//! also be set. These behaviors correspond to `exception_raise_state()`,
//! `exception_raise_state_identity()`, `mach_exception_raise_state()`, and
//! `mach_exception_raise_state_identity()`.
//!
//! \param[in] behavior An exception behavior value.
//!
//! \return `true` if \a behavior is `EXCEPTION_STATE` or
//!     `EXCEPTION_STATE_IDENTITY`, possibly with `MACH_EXCEPTION_CODES` also
//!      set.
bool ExceptionBehaviorHasState(exception_behavior_t behavior);

//! \brief Determines whether \a behavior indicates an exception behavior that
//!     carries thread and task identities.
//!
//! When this function returns `true`, an exception message of \a behavior will
//! carry thread and task identities in the form of send rights to the thread
//! and task ports. Its \a thread and \a task fields will be valid. When this
//! function returns `false`, these fields will not be valid.
//!
//! Exception behaviors that carry thread and task identity information are
//! `EXCEPTION_DEFAULT` and `EXCEPTION_STATE_IDENTITY`. `MACH_EXCEPTION_CODES`
//! may also be set. These behaviors correspond to `exception_raise()`,
//! `exception_raise_state_identity()`, `mach_exception_raise()`, and
//! `mach_exception_raise_state_identity()`.
//!
//! \param[in] behavior An exception behavior value.
//!
//! \return `true` if \a behavior is `EXCEPTION_DEFAULT` or
//!     `EXCEPTION_STATE_IDENTITY`, possibly with `MACH_EXCEPTION_CODES` also
//!      set.
bool ExceptionBehaviorHasIdentity(exception_behavior_t behavior);

//! \brief Determines whether \a behavior indicates an exception behavior that
//!     carries 64-bit exception codes (“Mach exception codes”).
//!
//! When this function returns `true`, an exception message of \a behavior will
//! carry 64-bit exception codes of type `mach_exception_code_t` in its \a code
//! field. When this function returns `false`, the exception message will carry
//! 32-bit exception codes of type `exception_data_type_t` in its \a code field.
//!
//! Exception behaviors that carry 64-bit exception codes are those that have
//! `MACH_EXCEPTION_CODES` set. These behaviors correspond to
//! `mach_exception_raise()`, `mach_exception_raise_state()`, and
//! `mach_exception_raise_state_identity()`.
//!
//! \param[in] behavior An exception behavior value.
//!
//! \return `true` if `MACH_EXCEPTION_CODES` is set in \a behavior.
bool ExceptionBehaviorHasMachExceptionCodes(exception_behavior_t behavior);

//! \brief Returns the basic behavior value of \a behavior, its value without
//!     `MACH_EXCEPTION_CODES` set.
//!
//! \param[in] behavior An exception behavior value.
//!
//! \return `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`, or
//!     `EXCEPTION_STATE_IDENTITY`, assuming \a behavior was a correct exception
//!     behavior value.
exception_behavior_t ExceptionBehaviorBasic(exception_behavior_t behavior);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_EXCEPTION_BEHAVIORS_H_
