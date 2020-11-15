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

#include "snapshot/linux/process_snapshot_linux.h"

#include <utility>

#include "base/logging.h"
#include "util/linux/exception_information.h"

namespace crashpad {

ProcessSnapshotLinux::ProcessSnapshotLinux() = default;

ProcessSnapshotLinux::~ProcessSnapshotLinux() = default;

bool ProcessSnapshotLinux::Initialize(PtraceConnection* connection) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (gettimeofday(&snapshot_time_, nullptr) != 0) {
    PLOG(ERROR) << "gettimeofday";
    return false;
  }

  if (!process_reader_.Initialize(connection) ||
      !memory_range_.Initialize(process_reader_.Memory(),
                                process_reader_.Is64Bit())) {
    return false;
  }

  system_.Initialize(&process_reader_, &snapshot_time_);

  InitializeThreads();
  InitializeModules();

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessSnapshotLinux::InitializeException(
    LinuxVMAddress exception_info_address) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  DCHECK(!exception_);

  ExceptionInformation info;
  if (!process_reader_.Memory()->Read(
          exception_info_address, sizeof(info), &info)) {
    LOG(ERROR) << "Couldn't read exception info";
    return false;
  }

  exception_.reset(new internal::ExceptionSnapshotLinux());
  if (!exception_->Initialize(&process_reader_,
                              info.siginfo_address,
                              info.context_address,
                              info.thread_id)) {
    exception_.reset();
    return false;
  }

  // The thread's existing snapshot will have captured the stack for the signal
  // handler. Replace it with a thread snapshot which captures the stack for the
  // exception context.
  for (const auto& reader_thread : process_reader_.Threads()) {
    if (reader_thread.tid == info.thread_id) {
      ProcessReaderLinux::Thread thread = reader_thread;
      thread.InitializeStackFromSP(&process_reader_,
                                   exception_->Context()->StackPointer());

      auto exc_thread_snapshot =
          std::make_unique<internal::ThreadSnapshotLinux>();
      if (!exc_thread_snapshot->Initialize(&process_reader_, thread)) {
        return false;
      }

      for (auto& thread_snapshot : threads_) {
        if (thread_snapshot->ThreadID() ==
            static_cast<uint64_t>(info.thread_id)) {
          thread_snapshot.reset(exc_thread_snapshot.release());
          return true;
        }
      }

      LOG(ERROR) << "thread not found " << info.thread_id;
      return false;
    }
  }

  LOG(ERROR) << "thread not found " << info.thread_id;
  return false;
}

void ProcessSnapshotLinux::GetCrashpadOptions(
    CrashpadInfoClientOptions* options) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  CrashpadInfoClientOptions local_options;

  for (const auto& module : modules_) {
    CrashpadInfoClientOptions module_options;
    if (!module->GetCrashpadOptions(&module_options)) {
      continue;
    }

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

pid_t ProcessSnapshotLinux::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_reader_.ProcessID();
}

pid_t ProcessSnapshotLinux::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_reader_.ParentProcessID();
}

void ProcessSnapshotLinux::SnapshotTime(timeval* snapshot_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *snapshot_time = snapshot_time_;
}

void ProcessSnapshotLinux::ProcessStartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  process_reader_.StartTime(start_time);
}

void ProcessSnapshotLinux::ProcessCPUTimes(timeval* user_time,
                                           timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  process_reader_.CPUTimes(user_time, system_time);
}

void ProcessSnapshotLinux::ReportID(UUID* report_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *report_id = report_id_;
}

void ProcessSnapshotLinux::ClientID(UUID* client_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *client_id = client_id_;
}

const std::map<std::string, std::string>&
ProcessSnapshotLinux::AnnotationsSimpleMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_simple_map_;
}

const SystemSnapshot* ProcessSnapshotLinux::System() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &system_;
}

std::vector<const ThreadSnapshot*> ProcessSnapshotLinux::Threads() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ThreadSnapshot*> threads;
  for (const auto& thread : threads_) {
    threads.push_back(thread.get());
  }
  return threads;
}

std::vector<const ModuleSnapshot*> ProcessSnapshotLinux::Modules() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> ProcessSnapshotLinux::UnloadedModules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(jperaza): Can this be implemented on Linux?
  return std::vector<UnloadedModuleSnapshot>();
}

const ExceptionSnapshot* ProcessSnapshotLinux::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_.get();
}

std::vector<const MemoryMapRegionSnapshot*> ProcessSnapshotLinux::MemoryMap()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(jperaza): do this.
  return std::vector<const MemoryMapRegionSnapshot*>();
}

std::vector<HandleSnapshot> ProcessSnapshotLinux::Handles() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<HandleSnapshot>();
}

std::vector<const MemorySnapshot*> ProcessSnapshotLinux::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<const MemorySnapshot*>();
}

void ProcessSnapshotLinux::InitializeThreads() {
  const std::vector<ProcessReaderLinux::Thread>& process_reader_threads =
      process_reader_.Threads();
  for (const ProcessReaderLinux::Thread& process_reader_thread :
       process_reader_threads) {
    auto thread = std::make_unique<internal::ThreadSnapshotLinux>();
    if (thread->Initialize(&process_reader_, process_reader_thread)) {
      threads_.push_back(std::move(thread));
    }
  }
}

void ProcessSnapshotLinux::InitializeModules() {
  for (const ProcessReaderLinux::Module& reader_module :
       process_reader_.Modules()) {
    auto module =
        std::make_unique<internal::ModuleSnapshotElf>(reader_module.name,
                                                      reader_module.elf_reader,
                                                      reader_module.type,
                                                      &memory_range_);
    if (module->Initialize()) {
      modules_.push_back(std::move(module));
    }
  }
}

}  // namespace crashpad
