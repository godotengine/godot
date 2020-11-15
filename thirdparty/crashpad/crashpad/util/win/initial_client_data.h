// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_WIN_INITIAL_CLIENT_DATA_H_
#define CRASHPAD_UTIL_WIN_INITIAL_CLIENT_DATA_H_

#include <windows.h>

#include <string>

#include "base/macros.h"
#include "util/win/address_types.h"

namespace crashpad {

//! \brief A container for the data associated with the `--initial-client-data`
//!     method for initializing the handler process on Windows.
class InitialClientData {
 public:
  //! \brief Constructs an unintialized instance to be used with
  //!     InitializeFromString().
  InitialClientData();

  //! \brief Constructs an instance of InitialClientData. This object does not
  //!     take ownership of any of the referenced HANDLEs.
  //!
  //! \param[in] request_crash_dump An event signalled from the client on crash.
  //! \param[in] request_non_crash_dump An event signalled from the client when
  //!     it would like a dump to be taken, but allowed to continue afterwards.
  //! \param[in] non_crash_dump_completed An event signalled from the handler to
  //!     tell the client that the non-crash dump has completed, and it can
  //!     continue execution.
  //! \param[in] first_pipe_instance The server end and first instance of a pipe
  //!     that will be used for communication with all other clients after this
  //!     initial one.
  //! \param[in] client_process A process handle for the client being
  //!     registered.
  //! \param[in] crash_exception_information The address, in the client's
  //!     address space, of an ExceptionInformation structure, used when
  //!     handling a crash dump request.
  //! \param[in] non_crash_exception_information The address, in the client's
  //!     address space, of an ExceptionInformation structure, used when
  //!     handling a non-crashing dump request.
  //! \param[in] debug_critical_section_address The address, in the client
  //!     process's address space, of a `CRITICAL_SECTION` allocated with a
  //!     valid .DebugInfo field. This can be accomplished by using
  //!     InitializeCriticalSectionWithDebugInfoIfPossible() or equivalent. This
  //!     value can be `0`, however then limited lock data will be available in
  //!     minidumps.
  InitialClientData(HANDLE request_crash_dump,
                    HANDLE request_non_crash_dump,
                    HANDLE non_crash_dump_completed,
                    HANDLE first_pipe_instance,
                    HANDLE client_process,
                    WinVMAddress crash_exception_information,
                    WinVMAddress non_crash_exception_information,
                    WinVMAddress debug_critical_section_address);

  //! \brief Returns whether the object has been initialized successfully.
  bool IsValid() const { return is_valid_; }

  //! Initializes this object from a string representation presumed to have been
  //!     created by StringRepresentation().
  //!
  //! \param[in] str The output of StringRepresentation().
  //!
  //! \return `true` on success, or `false` with a message logged on failure.
  bool InitializeFromString(const std::string& str);

  //! \brief Returns a string representation of the data of this object,
  //!     suitable for passing on the command line.
  std::string StringRepresentation() const;

  HANDLE request_crash_dump() const { return request_crash_dump_; }
  HANDLE request_non_crash_dump() const { return request_non_crash_dump_; }
  HANDLE non_crash_dump_completed() const { return non_crash_dump_completed_; }
  HANDLE first_pipe_instance() const { return first_pipe_instance_; }
  HANDLE client_process() const { return client_process_; }
  WinVMAddress crash_exception_information() const {
    return crash_exception_information_;
  }
  WinVMAddress non_crash_exception_information() const {
    return non_crash_exception_information_;
  }
  WinVMAddress debug_critical_section_address() const {
    return debug_critical_section_address_;
  }

 private:
  WinVMAddress crash_exception_information_;
  WinVMAddress non_crash_exception_information_;
  WinVMAddress debug_critical_section_address_;
  HANDLE request_crash_dump_;
  HANDLE request_non_crash_dump_;
  HANDLE non_crash_dump_completed_;
  HANDLE first_pipe_instance_;
  HANDLE client_process_;
  bool is_valid_;

  DISALLOW_COPY_AND_ASSIGN(InitialClientData);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_INITIAL_CLIENT_DATA_H_
