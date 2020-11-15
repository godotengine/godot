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

#ifndef CRASHPAD_UTIL_MACH_EXC_CLIENT_VARIANTS_H_
#define CRASHPAD_UTIL_MACH_EXC_CLIENT_VARIANTS_H_

#include <mach/mach.h>

#include "util/mach/mach_extensions.h"

namespace crashpad {

//! \brief Calls the appropriate `*exception_raise*()` function for the
//!     specified \a behavior.
//!
//! The function called will be `exception_raise()` for `EXCEPTION_DEFAULT`,
//! `exception_raise_state()` for `EXCEPTION_STATE`, or
//! `exception_raise_state_identity()` for `EXCEPTION_STATE_IDENTITY`. If
//! `MACH_EXCEPTION_CODES` is also set, the function called will instead be
//! `mach_exception_raise()`, `mach_exception_raise_state()` or
//! `mach_exception_raise_state_identity()`, respectively.
//!
//! This function does not fetch the existing thread state for \a behavior
//! values that require a thread state. The caller must provide the existing
//! thread state in the \a flavor, \a old_state, and \a old_state_count
//! parameters for \a behavior values that require a thread state. Thread states
//! may be obtained by calling `thread_get_state()` if needed. Similarly, this
//! function does not do anything with the new thread state returned for these
//! \a behavior values. Callers that wish to make use of the new thread state
//! may do so by using the returned \a flavor, \a new_state, and \a
//! new_state_count values. Thread states may be set by calling
//! `thread_set_state()` if needed.
//!
//! \a thread and \a task are only used when \a behavior indicates that the
//! exception message will carry identity information, when it has the value
//! `EXCEPTION_DEFAULT` or `EXCEPTION_STATE_IDENTITY`, possibly with
//! `MACH_EXCEPTION_CODES` also set. In other cases, these parameters are unused
//! and may be set to `THREAD_NULL` and `TASK_NULL`, respectively.
//!
//! \a flavor, \a old_state, \a old_state_count, \a new_state, and \a
//! new_state_count are only used when \a behavior indicates that the exception
//! message will carry thread state information, when it has the value
//! `EXCEPTION_STATE` or `EXCEPTION_STATE_IDENTITY`, possibly with
//! `MACH_EXCEPTION_CODES` also set. In other cases, these parameters are unused
//! and may be set to `0` (\a old_state_count) or `nullptr` (the remaining
//! parameters).
//!
//! Except as noted, the parameters and return value are equivalent to those of
//! the `*exception_raise*()` family of functions.
//!
//! \param[in] behavior The exception behavior, which dictates which function
//!     will be called. It is an error to call this function with an invalid
//!     value for \a behavior.
//! \param[in] exception_port
//! \param[in] thread
//! \param[in] task
//! \param[in] exception
//! \param[in] code If \a behavior indicates a behavior without
//!     `MACH_EXCEPTION_CODES`, the elements of \a code will be truncated in
//!     order to be passed to the appropriate exception handler.
//! \param[in] code_count
//! \param[in,out] flavor
//! \param[in] old_state
//! \param[in] old_state_count
//! \param[out] new_state
//! \param[out] new_state_count
//!
//! \return The return value of the function called.
kern_return_t UniversalExceptionRaise(exception_behavior_t behavior,
                                      exception_handler_t exception_port,
                                      thread_t thread,
                                      task_t task,
                                      exception_type_t exception,
                                      const mach_exception_data_type_t* code,
                                      mach_msg_type_number_t code_count,
                                      thread_state_flavor_t* flavor,
                                      ConstThreadState old_state,
                                      mach_msg_type_number_t old_state_count,
                                      thread_state_t new_state,
                                      mach_msg_type_number_t* new_state_count);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_EXC_CLIENT_VARIANTS_H_
