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

#ifndef CRASHPAD_UTIL_WIN_GET_FUNCTION_H_
#define CRASHPAD_UTIL_WIN_GET_FUNCTION_H_

#include <windows.h>

//! \file

namespace crashpad {
namespace internal {

//! \brief Returns a function pointer to a named function in a library.
//!
//! Do not call this directly, use the GET_FUNCTION() or GET_FUNCTION_REQUIRED()
//! macros instead.
//!
//! This accesses \a library by calling `LoadLibrary()` and is subject to the
//! same restrictions as that function. Notably, it can’t be used from a
//! `DllMain()` entry point.
//!
//! \param[in] library The library to search in.
//! \param[in] function The function to search for. If a leading `::` is
//!     present, it will be stripped.
//! \param[in] required If `true`, require the function to resolve by `DCHECK`.
//!
//! \return A pointer to the requested function on success. If \a required is
//!     `true`, triggers a `DCHECK` assertion on failure, otherwise, `nullptr`
//!     on failure.
FARPROC GetFunctionInternal(
    const wchar_t* library, const char* function, bool required);

//! \copydoc GetFunctionInternal
template <typename FunctionType>
FunctionType* GetFunction(
    const wchar_t* library, const char* function, bool required) {
  return reinterpret_cast<FunctionType*>(
      internal::GetFunctionInternal(library, function, required));
}

}  // namespace internal
}  // namespace crashpad

//! \brief Returns a function pointer to a named function in a library without
//!     requiring that it be found.
//!
//! If the library or function cannot be found, this will return `nullptr`. This
//! macro is intended to be used to access functions that may not be available
//! at runtime.
//!
//! This macro returns a properly-typed function pointer. It is expected to be
//! used in this way:
//! \code
//!     static const auto get_named_pipe_client_process_id =
//!         GET_FUNCTION(L"kernel32.dll", ::GetNamedPipeClientProcessId);
//!     if (get_named_pipe_client_process_id) {
//!       BOOL rv = get_named_pipe_client_process_id(pipe, &client_process_id);
//!     }
//! \endcode
//!
//! This accesses \a library by calling `LoadLibrary()` and is subject to the
//! same restrictions as that function. Notably, it can’t be used from a
//! `DllMain()` entry point.
//!
//! \param[in] library The library to search in.
//! \param[in] function The function to search for. A leading `::` is
//!     recommended when a wrapper function of the same name is present.
//!
//! \return A pointer to the requested function on success, or `nullptr` on
//!     failure.
//!
//! \sa GET_FUNCTION_REQUIRED
#define GET_FUNCTION(library, function)                  \
    crashpad::internal::GetFunction<decltype(function)>( \
        library, #function, false)

//! \brief Returns a function pointer to a named function in a library,
//!     requiring that it be found.
//!
//! If the library or function cannot be found, this will trigger a `DCHECK`
//! assertion. This macro is intended to be used to access functions that are
//! always expected to be available at runtime but which are not present in any
//! import library.
//!
//! This macro returns a properly-typed function pointer. It is expected to be
//! used in this way:
//! \code
//!     static const auto nt_query_object =
//!         GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtQueryObject);
//!     NTSTATUS status =
//!         nt_query_object(handle, type, &info, info_length, &return_length);
//! \endcode
//!
//! This accesses \a library by calling `LoadLibrary()` and is subject to the
//! same restrictions as that function. Notably, it can’t be used from a
//! `DllMain()` entry point.
//!
//! \param[in] library The library to search in.
//! \param[in] function The function to search for. A leading `::` is
//!     recommended when a wrapper function of the same name is present.
//!
//! \return A pointer to the requested function.
//!
//! \sa GET_FUNCTION
#define GET_FUNCTION_REQUIRED(library, function)         \
    crashpad::internal::GetFunction<decltype(function)>( \
        library, #function, true)

#endif  // CRASHPAD_UTIL_WIN_GET_FUNCTION_H_
