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

#include "snapshot/exception_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "snapshot/test/test_process_snapshot.h"

namespace crashpad {
namespace test {

TestProcessSnapshot::TestProcessSnapshot()
    : process_id_(0),
      parent_process_id_(0),
      snapshot_time_(),
      process_start_time_(),
      process_cpu_user_time_(),
      process_cpu_system_time_(),
      report_id_(),
      client_id_(),
      annotations_simple_map_(),
      system_(),
      threads_(),
      modules_(),
      exception_(),
      memory_map_(),
      handles_(),
      extra_memory_() {
}

TestProcessSnapshot::~TestProcessSnapshot() {
}

pid_t TestProcessSnapshot::ProcessID() const {
  return process_id_;
}

pid_t TestProcessSnapshot::ParentProcessID() const {
  return parent_process_id_;
}

void TestProcessSnapshot::SnapshotTime(timeval* snapshot_time) const {
  *snapshot_time = snapshot_time_;
}

void TestProcessSnapshot::ProcessStartTime(timeval* start_time) const {
  *start_time = process_start_time_;
}

void TestProcessSnapshot::ProcessCPUTimes(timeval* user_time,
                                          timeval* system_time) const {
  *user_time = process_cpu_user_time_;
  *system_time = process_cpu_system_time_;
}

void TestProcessSnapshot::ReportID(UUID* report_id) const {
  *report_id = report_id_;
}

void TestProcessSnapshot::ClientID(UUID* client_id) const {
  *client_id = client_id_;
}

const std::map<std::string, std::string>&
TestProcessSnapshot::AnnotationsSimpleMap() const {
  return annotations_simple_map_;
}

const SystemSnapshot* TestProcessSnapshot::System() const {
  return system_.get();
}

std::vector<const ThreadSnapshot*> TestProcessSnapshot::Threads() const {
  std::vector<const ThreadSnapshot*> threads;
  for (const auto& thread : threads_) {
    threads.push_back(thread.get());
  }
  return threads;
}

std::vector<const ModuleSnapshot*> TestProcessSnapshot::Modules() const {
  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> TestProcessSnapshot::UnloadedModules()
    const {
  return unloaded_modules_;
}

const ExceptionSnapshot* TestProcessSnapshot::Exception() const {
  return exception_.get();
}

std::vector<const MemoryMapRegionSnapshot*> TestProcessSnapshot::MemoryMap()
    const {
  std::vector<const MemoryMapRegionSnapshot*> memory_map;
  for (const auto& item : memory_map_) {
    memory_map.push_back(item.get());
  }
  return memory_map;
}

std::vector<HandleSnapshot> TestProcessSnapshot::Handles() const {
  return handles_;
}

std::vector<const MemorySnapshot*> TestProcessSnapshot::ExtraMemory() const {
  std::vector<const MemorySnapshot*> extra_memory;
  for (const auto& em : extra_memory_) {
    extra_memory.push_back(em.get());
  }
  return extra_memory;
}

}  // namespace test
}  // namespace crashpad
