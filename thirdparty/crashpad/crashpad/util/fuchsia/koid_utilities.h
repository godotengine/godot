// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_FUCHSIA_KOID_UTILITIES_H_
#define CRASHPAD_UTIL_FUCHSIA_KOID_UTILITIES_H_

#include <lib/zx/object.h>
#include <lib/zx/process.h>
#include <lib/zx/thread.h>
#include <zircon/syscalls/object.h>
#include <zircon/types.h>

#include <vector>

namespace crashpad {

//! \brief Get a list of child koids for a parent handle.
//!
//! For example, the list of processes in jobs, or the list of threads in a
//! process.
//!
//! \param[in] parent The handle to the parent object.
//! \param[in] child_kind The type of children to retrieve from \a parent. Valid
//!     values depend on the type of \a parent, but include
//!     `ZX_INFO_JOB_CHILDREN` (child jobs of a job), `ZX_INFO_JOB_PROCESSES`
//!     (child processes of a job), and `ZX_INFO_PROCESS_THREADS` (child threads
//!     of a process).
//! \return A vector of the koids representing the child objects.
//!
//! \sa GetChildHandles
std::vector<zx_koid_t> GetChildKoids(const zx::object_base& parent,
                                     zx_object_info_topic_t child_kind);

//! \brief Get handles representing a list of child objects of a given parent.
//!
//! \param[in] parent The handle to the parent object.
//! \return The resulting list of handles corresponding to the child objects.
//!
//! \sa GetChildKoids
std::vector<zx::thread> GetThreadHandles(const zx::process& parent);

//! \brief Convert a list of koids that are all children of a particular process
//!     into thread handles.
//!
//! \param[in] parent The parent object to which the koids belong.
//! \param[in] koids The list of koids.
//! \return The resulting list of handles corresponding to the koids. If an
//!     element of \a koids is invalid or can't be retrieved, there will be a
//!     corresponding `ZX_HANDLE_INVALID` entry in the return.
std::vector<zx::thread> GetHandlesForThreadKoids(
    const zx::process& parent,
    const std::vector<zx_koid_t>& koids);

//! \brief Retrieve the handle of a process' thread, based on koid.
//!
//! \param[in] parent The parent object to which the child belongs.
//! \param[in] child_koid The koid of the child to retrieve.
//! \return A handle representing \a child_koid, or `ZX_HANDLE_INVALID` if the
//!     handle could not be retrieved, in which case an error will be logged.
zx::thread GetThreadHandleByKoid(const zx::process& parent,
                                 zx_koid_t child_koid);

//! \brief Gets a process handle given the process' koid.
//!
//! \param[in] koid The process id.
//! \return A zx_handle_t (owned by a base::ScopedZxHandle) for the process. If
//!     the handle is invalid, an error will have been logged.
zx::process GetProcessFromKoid(zx_koid_t koid);

//! \brief Retrieves the koid for a given object handle.
//!
//! \param[in] object The handle for which the koid is to be retrieved.
//! \return The koid of \a handle, or `ZX_HANDLE_INVALID` with an error logged.
zx_koid_t GetKoidForHandle(const zx::object_base& object);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FUCHSIA_KOID_UTILITIES_H_
