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

#ifndef CRASHPAD_SNAPSHOT_EXCEPTION_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_EXCEPTION_SNAPSHOT_H_

#include <stdint.h>

#include <vector>

#include "snapshot/memory_snapshot.h"

namespace crashpad {

struct CPUContext;

//! \brief An abstract interface to a snapshot representing an exception that a
//!     snapshot process sustained and triggered the snapshot being taken.
class ExceptionSnapshot {
 public:
  virtual ~ExceptionSnapshot() {}

  //! \brief Returns a CPUContext object corresponding to the exception thread’s
  //!     CPU context at the time of the exception.
  //!
  //! The caller does not take ownership of this object, it is scoped to the
  //! lifetime of the ThreadSnapshot object that it was obtained from.
  virtual const CPUContext* Context() const = 0;

  //! \brief Returns the thread identifier of the thread that triggered the
  //!     exception.
  //!
  //! This value can be compared to ThreadSnapshot::ThreadID() to associate an
  //! ExceptionSnapshot object with the ThreadSnapshot that contains a snapshot
  //! of the thread that triggered the exception.
  virtual uint64_t ThreadID() const = 0;

  //! \brief Returns the top-level exception code identifying the exception.
  //!
  //! This is an operating system-specific value.
  //!
  //! For macOS, this will be an \ref EXC_x "EXC_*" exception type, such as
  //! `EXC_BAD_ACCESS`. `EXC_CRASH` will not appear here for exceptions
  //! processed as `EXC_CRASH` when generated from another preceding exception:
  //! the original exception code will appear instead. The exception type as it
  //! was received will appear at index 0 of Codes().
  //!
  //! For Windows, this will be an `EXCEPTION_*` exception type, such as
  //! `EXCEPTION_ACCESS_VIOLATION`.
  virtual uint32_t Exception() const = 0;

  //! \brief Returns the second-level exception code identifying the exception.
  //!
  //! This is an operating system-specific value.
  //!
  //! For macOS, this will be the value of the exception code at index 0 as
  //! received by a Mach exception handler, except:
  //!  * For `EXC_CRASH` exceptions generated from another preceding exception,
  //!    the original exception code will appear here, not the code as received
  //!    by the Mach exception handler.
  //!  * For `EXC_RESOURCE` and `EXC_GUARD` exceptions, the high 32 bits of the
  //!    exception code at index 0 will appear here.
  //!
  //! In all cases on macOS, the full exception code at index 0 as it was
  //! received will appear at index 1 of Codes().
  //!
  //! On Windows, this will either be `0` if the exception is continuable, or
  //! `EXCEPTION_NONCONTINUABLE` to indicate a noncontinuable exception.
  virtual uint32_t ExceptionInfo() const = 0;

  //! \brief Returns the address that triggered the exception.
  //!
  //! This may be the address that caused a fault on data access, or it may be
  //! the instruction pointer that contained an offending instruction. For
  //! exceptions where this value cannot be determined, it will be `0`.
  //!
  //! For macOS, this will be the value of the exception code at index 1 as
  //! received by a Mach exception handler.
  virtual uint64_t ExceptionAddress() const = 0;

  //! \brief Returns a series of operating system-specific exception codes.
  //!
  //! The precise interpretation of these codes is specific to the snapshot
  //! operating system. These codes may provide a duplicate of information
  //! available elsewhere, they may extend information available elsewhere, or
  //! they may not be present at all. In this case, an empty vector will be
  //! returned.
  //!
  //! For macOS, this will be a vector containing the original exception type
  //! and the values of `code[0]` and `code[1]` as received by a Mach exception
  //! handler.
  //!
  //! For Windows, these are additional arguments (if any) as provided to
  //! `RaiseException()`. See the documentation for `ExceptionInformation` in
  //! `EXCEPTION_RECORD`.
  virtual const std::vector<uint64_t>& Codes() const = 0;

  //! \brief Returns a vector of additional memory blocks that should be
  //!     included in a minidump.
  //!
  //! \return A vector of MemorySnapshot objects that will be included in the
  //!     crash dump. The caller does not take ownership of these objects, they
  //!     are scoped to the lifetime of the ThreadSnapshot object that they
  //!     were obtained from.
  virtual std::vector<const MemorySnapshot*> ExtraMemory() const = 0;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_EXCEPTION_SNAPSHOT_H_
