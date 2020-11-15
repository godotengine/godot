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

#ifndef CRASHPAD_SNAPSHOT_LINUX_PROCESS_SNAPSHOT_LINUX_H_
#define CRASHPAD_SNAPSHOT_LINUX_PROCESS_SNAPSHOT_LINUX_H_

#include <sys/time.h>
#include <sys/types.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/elf/module_snapshot_elf.h"
#include "snapshot/linux/exception_snapshot_linux.h"
#include "snapshot/linux/process_reader_linux.h"
#include "snapshot/linux/system_snapshot_linux.h"
#include "snapshot/linux/thread_snapshot_linux.h"
#include "snapshot/memory_map_region_snapshot.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/unloaded_module_snapshot.h"
#include "util/linux/ptrace_connection.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/uuid.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief A ProcessSnapshot of a running (or crashed) process running on a
//!     Linux system.
class ProcessSnapshotLinux final : public ProcessSnapshot {
 public:
  ProcessSnapshotLinux();
  ~ProcessSnapshotLinux() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] connection A connection to the process to snapshot.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(PtraceConnection* connection);

  //! \brief Initializes the object's exception.
  //!
  //! \param[in] exception_info The address of an ExceptionInformation in the
  //!     target process' address space.
  bool InitializeException(LinuxVMAddress exception_info);

  //! \brief Sets the value to be returned by ReportID().
  //!
  //! The crash report ID is under the control of the snapshot
  //! producer, which may call this method to set the report ID. If this is not
  //! done, ReportID() will return an identifier consisting entirely of zeroes.
  void SetReportID(const UUID& report_id) { report_id_ = report_id; }

  //! \brief Sets the value to be returned by ClientID().
  //!
  //! The client ID is under the control of the snapshot producer,
  //! which may call this method to set the client ID. If this is not done,
  //! ClientID() will return an identifier consisting entirely of zeroes.
  void SetClientID(const UUID& client_id) { client_id_ = client_id; }

  //! \brief Sets the value to be returned by AnnotationsSimpleMap().
  //!
  //! All process annotations are under the control of the snapshot
  //! producer, which may call this method to establish these annotations.
  //! Contrast this with module annotations, which are under the control of the
  //! process being snapshotted.
  void SetAnnotationsSimpleMap(
      const std::map<std::string, std::string>& annotations_simple_map) {
    annotations_simple_map_ = annotations_simple_map;
  }

  //! \brief Returns options from CrashpadInfo structures found in modules in
  //!     the process.
  //!
  //! \param[out] options Options set in CrashpadInfo structures in modules in
  //!     the process.
  void GetCrashpadOptions(CrashpadInfoClientOptions* options);

  // ProcessSnapshot:

  pid_t ProcessID() const override;
  pid_t ParentProcessID() const override;
  void SnapshotTime(timeval* snapshot_time) const override;
  void ProcessStartTime(timeval* start_time) const override;
  void ProcessCPUTimes(timeval* user_time, timeval* system_time) const override;
  void ReportID(UUID* report_id) const override;
  void ClientID(UUID* client_id) const override;
  const std::map<std::string, std::string>& AnnotationsSimpleMap()
      const override;
  const SystemSnapshot* System() const override;
  std::vector<const ThreadSnapshot*> Threads() const override;
  std::vector<const ModuleSnapshot*> Modules() const override;
  std::vector<UnloadedModuleSnapshot> UnloadedModules() const override;
  const ExceptionSnapshot* Exception() const override;
  std::vector<const MemoryMapRegionSnapshot*> MemoryMap() const override;
  std::vector<HandleSnapshot> Handles() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  void InitializeThreads();
  void InitializeModules();

  std::map<std::string, std::string> annotations_simple_map_;
  timeval snapshot_time_;
  UUID report_id_;
  UUID client_id_;
  std::vector<std::unique_ptr<internal::ThreadSnapshotLinux>> threads_;
  std::vector<std::unique_ptr<internal::ModuleSnapshotElf>> modules_;
  std::unique_ptr<internal::ExceptionSnapshotLinux> exception_;
  internal::SystemSnapshotLinux system_;
  ProcessReaderLinux process_reader_;
  ProcessMemoryRange memory_range_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessSnapshotLinux);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_PROCESS_SNAPSHOT_LINUX_H_
