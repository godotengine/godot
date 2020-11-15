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

#ifndef CRASHPAD_UTIL_MACH_MACH_EXTENSIONS_H_
#define CRASHPAD_UTIL_MACH_MACH_EXTENSIONS_H_

#include <mach/mach.h>

#include <string>

#include "base/mac/scoped_mach_port.h"

namespace crashpad {

//! \brief `MACH_PORT_NULL` with the correct type for a Mach port,
//!     `mach_port_t`.
//!
//! For situations where implicit conversions between signed and unsigned types
//! are not performed, use kMachPortNull instead of an explicit `implicit_cast`
//! of `MACH_PORT_NULL` to `mach_port_t`. This is useful for logging and testing
//! assertions.
constexpr mach_port_t kMachPortNull = MACH_PORT_NULL;

//! \brief `MACH_EXCEPTION_CODES` with the correct type for a Mach exception
//!     behavior, `exception_behavior_t`.
//!
//! Signedness problems can occur when ORing `MACH_EXCEPTION_CODES` as a signed
//! integer, because a signed integer overflow results. This constant can be
//! used instead of `MACH_EXCEPTION_CODES` in such cases.
constexpr exception_behavior_t kMachExceptionCodes = MACH_EXCEPTION_CODES;

// Because exception_mask_t is an int and has one bit for each defined
// exception_type_t, it’s reasonable to assume that there cannot be any
// officially-defined exception_type_t values higher than 31.
// kMachExceptionSimulated uses a value well outside this range because it does
// not require a corresponding mask value. Simulated exceptions are delivered to
// the exception handler registered for EXC_CRASH exceptions using
// EXC_MASK_CRASH.

//! \brief An exception type to use for simulated exceptions.
constexpr exception_type_t kMachExceptionSimulated = 'CPsx';

//! \brief A const version of `thread_state_t`.
//!
//! This is useful as the \a old_state parameter to exception handler functions.
//! Normally, these parameters are of type `thread_state_t`, but this allows
//! modification of the state, which is conceptually `const`.
using ConstThreadState = const natural_t*;

//! \brief Like `mach_thread_self()`, but without the obligation to release the
//!     send right.
//!
//! `mach_thread_self()` returns a send right to the current thread port,
//! incrementing its reference count. This burdens the caller with maintaining
//! this send right, and calling `mach_port_deallocate()` when it is no longer
//! needed. This is burdensome, and is at odds with the normal operation of
//! `mach_task_self()`, which does not increment the task port’s reference count
//! whose result must not be deallocated.
//!
//! Callers can use this function in preference to `mach_thread_self()`. This
//! function returns an extant reference to the current thread’s port without
//! incrementing its reference count.
//!
//! \return The value of `mach_thread_self()` without incrementing its reference
//!     count. The returned port must not be deallocated by
//!     `mach_port_deallocate()`. The returned value is valid as long as the
//!     thread continues to exist as a `pthread_t`.
thread_t MachThreadSelf();

//! \brief Creates a new Mach port in the current task.
//!
//! This function wraps the `mach_port_allocate()` providing a simpler
//! interface.
//!
//! \param[in] right The type of right to create.
//!
//! \return The new Mach port. On failure, `MACH_PORT_NULL` with a message
//!     logged.
mach_port_t NewMachPort(mach_port_right_t right);

//! \brief The value for `EXC_MASK_ALL` appropriate for the operating system at
//!     run time.
//!
//! The SDK’s definition of `EXC_MASK_ALL` has changed over time, with later
//! versions containing more bits set than earlier versions. However, older
//! kernels will reject exception masks that contain bits set that they don’t
//! recognize. Calling this function will return a value for `EXC_MASK_ALL`
//! appropriate for the system at run time.
//!
//! \note `EXC_MASK_ALL` does not include the value of `EXC_MASK_CRASH` or
//!     `EXC_MASK_CORPSE_NOTIFY`. Consumers that want `EXC_MASK_ALL` along with
//!     `EXC_MASK_CRASH` may use ExcMaskAll() `| EXC_MASK_CRASH`. Consumers may
//!     use ExcMaskValid() for `EXC_MASK_ALL` along with `EXC_MASK_CRASH`,
//!     `EXC_MASK_CORPSE_NOTIFY`, and any values that come into existence in the
//!     future.
exception_mask_t ExcMaskAll();

//! \brief An exception mask containing every possible exception understood by
//!     the operating system at run time.
//!
//! `EXC_MASK_ALL`, and thus ExcMaskAll(), never includes the value of
//! `EXC_MASK_CRASH` or `EXC_MASK_CORPSE_NOTIFY`. For situations where an
//! exception mask corresponding to every possible exception understood by the
//! running kernel is desired, use this function instead.
//!
//! Should new exception types be introduced in the future, this function will
//! be updated to include their bits in the returned mask value when run time
//! support is present.
exception_mask_t ExcMaskValid();

//! \brief Makes a `boostrap_check_in()` call to the process’ bootstrap server.
//!
//! This function is provided to make it easier to call `bootstrap_check_in()`
//! while avoiding accidental leaks of the returned receive right.
//!
//! \param[in] service_name The service name to check in.
//!
//! \return On success, a receive right to the service port. On failure,
//!     `MACH_PORT_NULL` with a message logged.
base::mac::ScopedMachReceiveRight BootstrapCheckIn(
    const std::string& service_name);

//! \brief Makes a `boostrap_look_up()` call to the process’ bootstrap server.
//!
//! This function is provided to make it easier to call `bootstrap_look_up()`
//! while avoiding accidental leaks of the returned send right.
//!
//! \param[in] service_name The service name to look up.
//!
//! \return On success, a send right to the service port. On failure,
//!     `MACH_PORT_NULL` with a message logged.
base::mac::ScopedMachSendRight BootstrapLookUp(const std::string& service_name);

//! \brief Obtains the system’s default Mach exception handler for crash-type
//!     exceptions.
//!
//! This is obtained by looking up `"com.apple.ReportCrash"` with the bootstrap
//! server. The service name comes from the first launch agent loaded by
//! `launchd` with a `MachServices` entry having `ExceptionServer` set. This
//! launch agent is normally loaded from
//! `/System/Library/LaunchAgents/com.apple.ReportCrash.plist`.
//!
//! \return On success, a send right to an `exception_handler_t` corresponding
//!     to the system’s default crash reporter. On failure, `MACH_PORT_NULL`,
//!     with a message logged.
base::mac::ScopedMachSendRight SystemCrashReporterHandler();

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_MACH_EXTENSIONS_H_
