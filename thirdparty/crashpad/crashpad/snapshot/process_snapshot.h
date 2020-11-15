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

#ifndef CRASHPAD_SNAPSHOT_PROCESS_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_PROCESS_SNAPSHOT_H_

#include <sys/time.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "snapshot/handle_snapshot.h"
#include "util/misc/uuid.h"

namespace crashpad {

class ExceptionSnapshot;
class MemoryMapRegionSnapshot;
class MemorySnapshot;
class ModuleSnapshot;
class SystemSnapshot;
class ThreadSnapshot;
class UnloadedModuleSnapshot;

//! \brief An abstract interface to a snapshot representing the state of a
//!     process.
//!
//! This is the top-level object in a family of Snapshot objects, because it
//! gives access to a SystemSnapshot, vectors of ModuleSnapshot and
//! ThreadSnapshot objects, and possibly an ExceptionSnapshot. In turn,
//! ThreadSnapshot and ExceptionSnapshot objects both give access to CPUContext
//! objects, and ThreadSnapshot objects also give access to MemorySnapshot
//! objects corresponding to thread stacks.
class ProcessSnapshot {
 public:
  virtual ~ProcessSnapshot() {}

  //! \brief Returns the snapshot process’ process ID.
  virtual pid_t ProcessID() const = 0;

  //! \brief Returns the snapshot process’ parent process’ process ID.
  virtual pid_t ParentProcessID() const = 0;

  //! \brief Returns the time that the snapshot was taken in \a snapshot_time.
  //!
  //! \param[out] snapshot_time The time that the snapshot was taken. This is
  //!     distinct from the time that a ProcessSnapshot object was created or
  //!     initialized, although it may be that time for ProcessSnapshot objects
  //!     representing live or recently-crashed process state.
  virtual void SnapshotTime(timeval* snapshot_time) const = 0;

  //! \brief Returns the time that the snapshot process was started in \a
  //!     start_time.
  //!
  //! Normally, process uptime in wall clock time can be computed as
  //! SnapshotTime() − ProcessStartTime(), but this cannot be guaranteed in
  //! cases where the real-time clock has been set during the snapshot process’
  //! lifetime.
  //!
  //! \param[out] start_time The time that the process was started.
  virtual void ProcessStartTime(timeval* start_time) const = 0;

  //! \brief Returns the snapshot process’ CPU usage times in \a user_time and
  //!     \a system_time.
  //!
  //! \param[out] user_time The time that the process has spent executing in
  //!     user mode.
  //! \param[out] system_time The time that the process has spent executing in
  //!     system (kernel) mode.
  virtual void ProcessCPUTimes(timeval* user_time,
                               timeval* system_time) const = 0;

  //! \brief Returns a %UUID identifying the event that the snapshot describes.
  //!
  //! This provides a stable identifier for a crash even as the report is
  //! converted to different formats, provided that all formats support storing
  //! a crash report ID. When a report is originally created, a report ID should
  //! be assigned. From that point on, any operations involving the same report
  //! should preserve the same report ID.
  //!
  //! If no identifier is available, this field will contain zeroes.
  virtual void ReportID(UUID* client_id) const = 0;

  //! \brief Returns a %UUID identifying the client that the snapshot
  //!     represents.
  //!
  //! Client identification is within the scope of the application, but it is
  //! expected that the identifier will be unique for an instance of Crashpad
  //! monitoring an application or set of applications for a user. The
  //! identifier shall remain stable over time.
  //!
  //! If no identifier is available, this field will contain zeroes.
  virtual void ClientID(UUID* client_id) const = 0;

  //! \brief Returns key-value string annotations recorded for the process,
  //!     system, or snapshot producer.
  //!
  //! This method retrieves annotations recorded for a process. These
  //! annotations are intended for diagnostic use, including crash analysis.
  //! “Simple annotations” are structured as a sequence of key-value pairs,
  //! where all keys and values are strings. These are referred to in Chrome as
  //! “crash keys.”
  //!
  //! Annotations stored here may reflect the process, system, or snapshot
  //! producer. Most annotations not under the client’s direct control will be
  //! retrievable by this method. For clients such as Chrome, this includes the
  //! product name and version.
  //!
  //! Additional per-module annotations may be obtained by calling
  //! ModuleSnapshot::AnnotationsSimpleMap().
  //
  // This interface currently returns a const& because all implementations store
  // the data within their objects in this format, and are therefore able to
  // provide this access without requiring a copy. Future implementations may
  // obtain data on demand or store data in a different format, at which point
  // the cost of maintaining this data in ProcessSnapshot subclass objects will
  // need to be taken into account, and this interface possibly revised.
  virtual const std::map<std::string, std::string>& AnnotationsSimpleMap()
      const = 0;

  //! \brief Returns a SystemSnapshot reflecting the characteristics of the
  //!     system that ran the snapshot process at the time of the snapshot.
  //!
  //! \return A SystemSnapshot object. The caller does not take ownership of
  //!     this object, it is scoped to the lifetime of the ProcessSnapshot
  //!     object that it was obtained from.
  virtual const SystemSnapshot* System() const = 0;

  //! \brief Returns ModuleSnapshot objects reflecting the code modules (binary
  //!     images) loaded into the snapshot process at the time of the snapshot.
  //!
  //! \return A vector of ModuleSnapshot objects. The caller does not take
  //!     ownership of these objects, they are scoped to the lifetime of the
  //!     ProcessSnapshot object that they were obtained from.
  virtual std::vector<const ModuleSnapshot*> Modules() const = 0;

  //! \brief Returns UnloadedModuleSnapshot objects reflecting the code modules
  //!     the were recorded as unloaded at the time of the snapshot.
  //!
  //! \return A vector of UnloadedModuleSnapshot objects.
  virtual std::vector<UnloadedModuleSnapshot> UnloadedModules() const = 0;

  //! \brief Returns ThreadSnapshot objects reflecting the threads (lightweight
  //!     processes) existing in the snapshot process at the time of the
  //!     snapshot.
  //!
  //! \return A vector of ThreadSnapshot objects. The caller does not take
  //!     ownership of these objects, they are scoped to the lifetime of the
  //!     ProcessSnapshot object that they were obtained from.
  virtual std::vector<const ThreadSnapshot*> Threads() const = 0;

  //! \brief Returns an ExceptionSnapshot reflecting the exception that the
  //!     snapshot process sustained to trigger the snapshot being taken.
  //!
  //! \return An ExceptionSnapshot object. The caller does not take ownership of
  //!     this object, it is scoped to the lifetime of the ProcessSnapshot
  //!     object that it was obtained from. If the snapshot is not a result of
  //!     an exception, returns `nullptr`.
  virtual const ExceptionSnapshot* Exception() const = 0;

  //! \brief Returns MemoryMapRegionSnapshot objects reflecting the regions
  //!     of the memory map in the snapshot process at the time of the snapshot.
  //!
  //! \return A vector of MemoryMapRegionSnapshot objects. The caller does not
  //!     take ownership of these objects, they are scoped to the lifetime of
  //!     the ProcessSnapshot object that they were obtained from.
  virtual std::vector<const MemoryMapRegionSnapshot*> MemoryMap() const = 0;

  //! \brief Returns HandleSnapshot objects reflecting the open handles in the
  //!     snapshot process at the time of the snapshot.
  //!
  //! \return A vector of HandleSnapshot objects.
  virtual std::vector<HandleSnapshot> Handles() const = 0;

  //! \brief Returns a vector of additional memory blocks that should be
  //!     included in a minidump.
  //!
  //! \return An vector of MemorySnapshot objects that will be included in the
  //!     crash dump. The caller does not take ownership of these objects, they
  //!     are scoped to the lifetime of the ProcessSnapshot object that they
  //!     were obtained from.
  virtual std::vector<const MemorySnapshot*> ExtraMemory() const = 0;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_PROCESS_SNAPSHOT_H_
