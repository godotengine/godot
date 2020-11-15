// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_LINUX_PTRACER_H_
#define CRASHPAD_UTIL_LINUX_PTRACER_H_

#include <sys/types.h>

#include "base/macros.h"
#include "util/linux/address_types.h"
#include "util/linux/thread_info.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief Provides an architecturally agnostic interface for collecting
//!     information with `ptrace`.
//!
//! A ptracer is configured for a particular bitness. It is an error to make any
//! calls via this object against a thread whose bitness does not match the
//! bitness this object was initialized with.
class Ptracer {
 public:
  //! \brief Constructs this object with a pre-determined bitness.
  //!
  //! \param[in] is_64_bit `true` if this object is to be configured for 64-bit.
  //! \param[in] can_log Whether methods in this class can log error messages.
  Ptracer(bool is_64_bit, bool can_log);

  //! \brief Constructs this object without a pre-determined bitness.
  //!
  //! Initialize() must be successfully called before making any other calls on
  //! this object.
  //!
  //! \param[in] can_log Whether methods in this class can log error messages.
  explicit Ptracer(bool can_log);

  ~Ptracer();

  //! \brief Initializes this object to the bitness of the process whose process
  //!     ID is \a pid.
  //!
  //! \param[in] pid The process ID of the process to initialize with.
  //! \return `true` on success. `false` on failure with a message logged, if
  //!     enabled.
  bool Initialize(pid_t pid);

  //! \brief Return `true` if this object is configured for 64-bit.
  bool Is64Bit();

  //! \brief Uses `ptrace` to collect information about the thread with thread
  //!     ID \a tid.
  //!
  //! The target thread should be attached before calling this method.
  //! \see ScopedPtraceAttach
  //!
  //! \param[in] tid The thread ID of the thread to collect information for.
  //! \param[out] info A ThreadInfo for the thread.
  //! \return `true` on success. `false` on failure with a message logged, if
  //!     enabled.
  bool GetThreadInfo(pid_t tid, ThreadInfo* info);

  //! \brief Uses `ptrace` to read memory from the process with process ID \a
  //!     pid, up to a maximum number of bytes.
  //!
  //! The target process should already be attached before calling this method.
  //! \see ScopedPtraceAttach
  //!
  //! \param[in] pid The process ID whose memory to read.
  //! \param[in] address The base address of the region to read.
  //! \param[in] size The size of the memory region to read. \a buffer must be
  //!     at least this size.
  //! \param[out] buffer The buffer to fill with the data read.
  //! \return the number of bytes read, 0 if there are no more bytes to read, or
  //!     -1 on failure with a message logged if logging is enabled.
  ssize_t ReadUpTo(pid_t pid,
                   LinuxVMAddress address,
                   size_t size,
                   char* buffer);

 private:
  ssize_t ReadLastBytes(pid_t pid,
                        LinuxVMAddress address,
                        size_t size,
                        char* buffer);

  bool is_64_bit_;
  bool can_log_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(Ptracer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_PTRACER_H_
