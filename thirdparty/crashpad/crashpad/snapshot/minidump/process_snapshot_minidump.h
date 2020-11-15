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

#ifndef CRASHPAD_SNAPSHOT_MINIDUMP_PROCESS_SNAPSHOT_MINIDUMP_H_
#define CRASHPAD_SNAPSHOT_MINIDUMP_PROCESS_SNAPSHOT_MINIDUMP_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/time.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/minidump/module_snapshot_minidump.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/unloaded_module_snapshot.h"
#include "util/file/file_reader.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/uuid.h"

namespace crashpad {

//! \brief A ProcessSnapshot based on a minidump file.
class ProcessSnapshotMinidump final : public ProcessSnapshot {
 public:
  ProcessSnapshotMinidump();
  ~ProcessSnapshotMinidump() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] file_reader A file reader corresponding to a minidump file.
  //!     The file reader must support seeking.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(FileReaderInterface* file_reader);

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
  // Initializes data carried in a MinidumpCrashpadInfo stream on behalf of
  // Initialize().
  bool InitializeCrashpadInfo();

  // Initializes data carried in a MINIDUMP_MODULE_LIST stream on behalf of
  // Initialize().
  bool InitializeModules();

  // Initializes data carried in a MinidumpModuleCrashpadInfoList structure on
  // behalf of InitializeModules(). This makes use of MinidumpCrashpadInfo as
  // well, so it must be called after InitializeCrashpadInfo().
  bool InitializeModulesCrashpadInfo(
      std::map<uint32_t, MINIDUMP_LOCATION_DESCRIPTOR>*
          module_crashpad_info_links);

  // Initializes data carried in a MINIDUMP_MISC_INFO structure on behalf of
  // Initialize().
  bool InitializeMiscInfo();

  MINIDUMP_HEADER header_;
  std::vector<MINIDUMP_DIRECTORY> stream_directory_;
  std::map<MinidumpStreamType, const MINIDUMP_LOCATION_DESCRIPTOR*> stream_map_;
  std::vector<std::unique_ptr<internal::ModuleSnapshotMinidump>> modules_;
  std::vector<UnloadedModuleSnapshot> unloaded_modules_;
  MinidumpCrashpadInfo crashpad_info_;
  std::map<std::string, std::string> annotations_simple_map_;
  FileReaderInterface* file_reader_;  // weak
  pid_t process_id_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessSnapshotMinidump);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MINIDUMP_PROCESS_SNAPSHOT_MINIDUMP_H_
