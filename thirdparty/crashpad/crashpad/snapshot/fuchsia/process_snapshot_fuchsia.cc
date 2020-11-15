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

#include "snapshot/fuchsia/process_snapshot_fuchsia.h"

#include "base/logging.h"
#include "util/fuchsia/koid_utilities.h"

namespace crashpad {

ProcessSnapshotFuchsia::ProcessSnapshotFuchsia() = default;

ProcessSnapshotFuchsia::~ProcessSnapshotFuchsia() = default;

bool ProcessSnapshotFuchsia::Initialize(const zx::process& process) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (gettimeofday(&snapshot_time_, nullptr) != 0) {
    PLOG(ERROR) << "gettimeofday";
    return false;
  }

  if (!process_reader_.Initialize(process) ||
      !memory_range_.Initialize(process_reader_.Memory(), true)) {
    return false;
  }

  system_.Initialize(&snapshot_time_);

  InitializeThreads();
  InitializeModules();

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessSnapshotFuchsia::InitializeException(
    zx_koid_t thread_id,
    const zx_exception_report_t& report) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  exception_.reset(new internal::ExceptionSnapshotFuchsia());
  exception_->Initialize(&process_reader_, thread_id, report);
  return true;
}

void ProcessSnapshotFuchsia::GetCrashpadOptions(
    CrashpadInfoClientOptions* options) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  CrashpadInfoClientOptions local_options;

  for (const auto& module : modules_) {
    CrashpadInfoClientOptions module_options;
    module->GetCrashpadOptions(&module_options);

    if (local_options.crashpad_handler_behavior == TriState::kUnset) {
      local_options.crashpad_handler_behavior =
          module_options.crashpad_handler_behavior;
    }
    if (local_options.system_crash_reporter_forwarding == TriState::kUnset) {
      local_options.system_crash_reporter_forwarding =
          module_options.system_crash_reporter_forwarding;
    }
    if (local_options.gather_indirectly_referenced_memory == TriState::kUnset) {
      local_options.gather_indirectly_referenced_memory =
          module_options.gather_indirectly_referenced_memory;
      local_options.indirectly_referenced_memory_cap =
          module_options.indirectly_referenced_memory_cap;
    }

    // If non-default values have been found for all options, the loop can end
    // early.
    if (local_options.crashpad_handler_behavior != TriState::kUnset &&
        local_options.system_crash_reporter_forwarding != TriState::kUnset &&
        local_options.gather_indirectly_referenced_memory != TriState::kUnset) {
      break;
    }
  }

  *options = local_options;
}

pid_t ProcessSnapshotFuchsia::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return GetKoidForHandle(*zx::process::self());
}

pid_t ProcessSnapshotFuchsia::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196
  return 0;
}

void ProcessSnapshotFuchsia::SnapshotTime(timeval* snapshot_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *snapshot_time = snapshot_time_;
}

void ProcessSnapshotFuchsia::ProcessStartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): https://crashpad.chromium.org/bug/196. Nothing available.
  *start_time = timeval{};
}

void ProcessSnapshotFuchsia::ProcessCPUTimes(timeval* user_time,
                                             timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): https://crashpad.chromium.org/bug/196. Nothing available.
  *user_time = timeval{};
  *system_time = timeval{};
}

void ProcessSnapshotFuchsia::ReportID(UUID* report_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *report_id = report_id_;
}

void ProcessSnapshotFuchsia::ClientID(UUID* client_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *client_id = client_id_;
}

const std::map<std::string, std::string>&
ProcessSnapshotFuchsia::AnnotationsSimpleMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_simple_map_;
}

const SystemSnapshot* ProcessSnapshotFuchsia::System() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &system_;
}

std::vector<const ThreadSnapshot*> ProcessSnapshotFuchsia::Threads() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ThreadSnapshot*> threads;
  for (const auto& thread : threads_) {
    threads.push_back(thread.get());
  }
  return threads;
}

std::vector<const ModuleSnapshot*> ProcessSnapshotFuchsia::Modules() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> ProcessSnapshotFuchsia::UnloadedModules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // dlclose() never unloads on Fuchsia. ZX-1728 upstream.
  return std::vector<UnloadedModuleSnapshot>();
}

const ExceptionSnapshot* ProcessSnapshotFuchsia::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_.get();
}

std::vector<const MemoryMapRegionSnapshot*> ProcessSnapshotFuchsia::MemoryMap()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<const MemoryMapRegionSnapshot*>();
}

std::vector<HandleSnapshot> ProcessSnapshotFuchsia::Handles() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<HandleSnapshot>();
}

std::vector<const MemorySnapshot*> ProcessSnapshotFuchsia::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<const MemorySnapshot*>();
}

void ProcessSnapshotFuchsia::InitializeThreads() {
  const std::vector<ProcessReaderFuchsia::Thread>& process_reader_threads =
      process_reader_.Threads();
  for (const ProcessReaderFuchsia::Thread& process_reader_thread :
       process_reader_threads) {
    auto thread = std::make_unique<internal::ThreadSnapshotFuchsia>();
    if (thread->Initialize(&process_reader_, process_reader_thread)) {
      threads_.push_back(std::move(thread));
    }
  }
}

void ProcessSnapshotFuchsia::InitializeModules() {
  for (const ProcessReaderFuchsia::Module& reader_module :
       process_reader_.Modules()) {
    auto module =
        std::make_unique<internal::ModuleSnapshotElf>(reader_module.name,
                                                      reader_module.reader,
                                                      reader_module.type,
                                                      &memory_range_);
    if (module->Initialize()) {
      modules_.push_back(std::move(module));
    }
  }
}

}  // namespace crashpad
