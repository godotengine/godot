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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_PROCESS_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_PROCESS_SNAPSHOT_H_

#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/macros.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/memory_map_region_snapshot.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/unloaded_module_snapshot.h"
#include "util/misc/uuid.h"

namespace crashpad {
namespace test {

//! \brief A test ProcessSnapshot that can carry arbitrary data for testing
//!     purposes.
class TestProcessSnapshot final : public ProcessSnapshot {
 public:
  TestProcessSnapshot();
  ~TestProcessSnapshot() override;

  void SetProcessID(pid_t process_id) { process_id_ = process_id; }
  void SetParentProcessID(pid_t parent_process_id) {
    parent_process_id_ = parent_process_id;
  }
  void SetSnapshotTime(const timeval& snapshot_time) {
    snapshot_time_ = snapshot_time;
  }
  void SetProcessStartTime(const timeval& start_time) {
    process_start_time_ = start_time;
  }
  void SetProcessCPUTimes(const timeval& user_time,
                          const timeval& system_time) {
    process_cpu_user_time_ = user_time;
    process_cpu_system_time_ = system_time;
  }
  void SetReportID(const UUID& report_id) { report_id_ = report_id; }
  void SetClientID(const UUID& client_id) { client_id_ = client_id; }
  void SetAnnotationsSimpleMap(
      const std::map<std::string, std::string>& annotations_simple_map) {
    annotations_simple_map_ = annotations_simple_map;
  }

  //! \brief Sets the system snapshot to be returned by System().
  //!
  //! \param[in] system The system snapshot that System() will return. The
  //!     TestProcessSnapshot object takes ownership of \a system.
  void SetSystem(std::unique_ptr<SystemSnapshot> system) {
    system_ = std::move(system);
  }

  //! \brief Adds a thread snapshot to be returned by Threads().
  //!
  //! \param[in] thread The thread snapshot that will be included in Threads().
  //!     The TestProcessSnapshot object takes ownership of \a thread.
  void AddThread(std::unique_ptr<ThreadSnapshot> thread) {
    threads_.push_back(std::move(thread));
  }

  //! \brief Adds a module snapshot to be returned by Modules().
  //!
  //! \param[in] module The module snapshot that will be included in Modules().
  //!     The TestProcessSnapshot object takes ownership of \a module.
  void AddModule(std::unique_ptr<ModuleSnapshot> module) {
    modules_.push_back(std::move(module));
  }

  //! \brief Adds an unloaded module snapshot to be returned by
  //!     UnloadedModules().
  //!
  //! \param[in] unloaded_module The unloaded module snapshot that will be
  //!     included in UnloadedModules().
  void AddModule(const UnloadedModuleSnapshot& unloaded_module) {
    unloaded_modules_.push_back(unloaded_module);
  }

  //! \brief Sets the exception snapshot to be returned by Exception().
  //!
  //! \param[in] exception The exception snapshot that Exception() will return.
  //!     The TestProcessSnapshot object takes ownership of \a exception.
  void SetException(std::unique_ptr<ExceptionSnapshot> exception) {
    exception_ = std::move(exception);
  }

  //! \brief Adds a memory map region snapshot to be returned by MemoryMap().
  //!
  //! \param[in] region The memory map region snapshot that will be included in
  //!     MemoryMap(). The TestProcessSnapshot object takes ownership of \a
  //!     region.
  void AddMemoryMapRegion(std::unique_ptr<MemoryMapRegionSnapshot> region) {
    memory_map_.push_back(std::move(region));
  }

  //! \brief Adds a handle snapshot to be returned by Handles().
  //!
  //! \param[in] handle The handle snapshot that will be included in Handles().
  void AddHandle(const HandleSnapshot& handle) {
    handles_.push_back(handle);
  }

  //! \brief Add a memory snapshot to be returned by ExtraMemory().
  //!
  //! \param[in] extra_memory The memory snapshot that will be included in
  //!     ExtraMemory(). The TestProcessSnapshot object takes ownership of \a
  //!     extra_memory.
  void AddExtraMemory(std::unique_ptr<MemorySnapshot> extra_memory) {
    extra_memory_.push_back(std::move(extra_memory));
  }

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
  pid_t process_id_;
  pid_t parent_process_id_;
  timeval snapshot_time_;
  timeval process_start_time_;
  timeval process_cpu_user_time_;
  timeval process_cpu_system_time_;
  UUID report_id_;
  UUID client_id_;
  std::map<std::string, std::string> annotations_simple_map_;
  std::unique_ptr<SystemSnapshot> system_;
  std::vector<std::unique_ptr<ThreadSnapshot>> threads_;
  std::vector<std::unique_ptr<ModuleSnapshot>> modules_;
  std::vector<UnloadedModuleSnapshot> unloaded_modules_;
  std::unique_ptr<ExceptionSnapshot> exception_;
  std::vector<std::unique_ptr<MemoryMapRegionSnapshot>> memory_map_;
  std::vector<HandleSnapshot> handles_;
  std::vector<std::unique_ptr<MemorySnapshot>> extra_memory_;

  DISALLOW_COPY_AND_ASSIGN(TestProcessSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_PROCESS_SNAPSHOT_H_
