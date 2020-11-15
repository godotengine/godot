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

#ifndef CRASHPAD_SNAPSHOT_MAC_PROCESS_SNAPSHOT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_PROCESS_SNAPSHOT_MAC_H_

#include <mach/mach.h>
#include <sys/time.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "client/crashpad_info.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/mac/exception_snapshot_mac.h"
#include "snapshot/mac/module_snapshot_mac.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/mac/system_snapshot_mac.h"
#include "snapshot/mac/thread_snapshot_mac.h"
#include "snapshot/memory_map_region_snapshot.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/unloaded_module_snapshot.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/uuid.h"

namespace crashpad {

//! \brief A ProcessSnapshot of a running (or crashed) process running on a
//!     macOS system.
class ProcessSnapshotMac final : public ProcessSnapshot {
 public:
  ProcessSnapshotMac();
  ~ProcessSnapshotMac() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] task The task to create a snapshot from.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(task_t task);

  //! \brief Initializes the object’s exception.
  //!
  //! This populates the data to be returned by Exception(). The parameters may
  //! be passed directly through from a Mach exception handler.
  //!
  //! This method must not be called until after a successful call to
  //! Initialize().
  //!
  //! \return `true` if the exception information could be initialized, `false`
  //!     otherwise with an appropriate message logged. When this method returns
  //!     `false`, the ProcessSnapshotMac object’s validity remains unchanged.
  bool InitializeException(exception_behavior_t behavior,
                           thread_t exception_thread,
                           exception_type_t exception,
                           const mach_exception_data_type_t* code,
                           mach_msg_type_number_t code_count,
                           thread_state_flavor_t flavor,
                           ConstThreadState state,
                           mach_msg_type_number_t state_count);

  //! \brief Sets the value to be returned by ReportID().
  //!
  //! On macOS, the crash report ID is under the control of the snapshot
  //! producer, which may call this method to set the report ID. If this is not
  //! done, ReportID() will return an identifier consisting entirely of zeroes.
  void SetReportID(const UUID& report_id) { report_id_ = report_id; }

  //! \brief Sets the value to be returned by ClientID().
  //!
  //! On macOS, the client ID is under the control of the snapshot producer,
  //! which may call this method to set the client ID. If this is not done,
  //! ClientID() will return an identifier consisting entirely of zeroes.
  void SetClientID(const UUID& client_id) { client_id_ = client_id; }

  //! \brief Sets the value to be returned by AnnotationsSimpleMap().
  //!
  //! On macOS, all process annotations are under the control of the snapshot
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
  // Initializes threads_ on behalf of Initialize().
  void InitializeThreads();

  // Initializes modules_ on behalf of Initialize().
  void InitializeModules();

  internal::SystemSnapshotMac system_;
  std::vector<std::unique_ptr<internal::ThreadSnapshotMac>> threads_;
  std::vector<std::unique_ptr<internal::ModuleSnapshotMac>> modules_;
  std::unique_ptr<internal::ExceptionSnapshotMac> exception_;
  ProcessReaderMac process_reader_;
  UUID report_id_;
  UUID client_id_;
  std::map<std::string, std::string> annotations_simple_map_;
  timeval snapshot_time_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessSnapshotMac);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_PROCESS_SNAPSHOT_MAC_H_
