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

#ifndef CRASHPAD_SNAPSHOT_THREAD_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_THREAD_SNAPSHOT_H_

#include <stdint.h>

#include <vector>

namespace crashpad {

struct CPUContext;
class MemorySnapshot;

//! \brief An abstract interface to a snapshot representing a thread
//!     (lightweight process) present in a snapshot process.
class ThreadSnapshot {
 public:
  virtual ~ThreadSnapshot() {}

  //! \brief Returns a CPUContext object corresponding to the thread’s CPU
  //!     context.
  //!
  //! The caller does not take ownership of this object, it is scoped to the
  //! lifetime of the ThreadSnapshot object that it was obtained from.
  virtual const CPUContext* Context() const = 0;

  //! \brief Returns a MemorySnapshot object corresponding to the memory region
  //!     that contains the thread’s stack, or `nullptr` if no stack region is
  //!     available.
  //!
  //! The caller does not take ownership of this object, it is scoped to the
  //! lifetime of the ThreadSnapshot object that it was obtained from.
  virtual const MemorySnapshot* Stack() const = 0;

  //! \brief Returns the thread’s identifier.
  //!
  //! %Thread identifiers are at least unique within a process, and may be
  //! unique system-wide.
  virtual uint64_t ThreadID() const = 0;

  //! \brief Returns the thread’s suspend count.
  //!
  //! A suspend count of `0` denotes a schedulable (not suspended) thread.
  virtual int SuspendCount() const = 0;

  //! \brief Returns the thread’s priority.
  //!
  //! Threads with higher priorities will have higher priority values.
  virtual int Priority() const = 0;

  //! \brief Returns the base address of a region used to store thread-specific
  //!     data.
  virtual uint64_t ThreadSpecificDataAddress() const = 0;

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

#endif  // CRASHPAD_SNAPSHOT_THREAD_SNAPSHOT_H_
