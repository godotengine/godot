// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "snapshot/sanitized/process_snapshot_sanitized.h"

#include <stdint.h>

#include "snapshot/cpu_context.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

namespace {

class StackReferencesAddressRange : public MemorySnapshot::Delegate {
 public:
  // Returns true if stack contains a pointer aligned word in the range [low,
  // high). The search starts at the first pointer aligned address greater than
  // stack_pointer.
  bool CheckStack(const MemorySnapshot* stack,
                  VMAddress stack_pointer,
                  VMAddress low,
                  VMAddress high,
                  bool is_64_bit) {
    stack_ = stack;
    stack_pointer_ = stack_pointer;
    low_ = low;
    high_ = high;
    is_64_bit_ = is_64_bit;
    return stack_->Read(this);
  }

  // MemorySnapshot::Delegate
  bool MemorySnapshotDelegateRead(void* data, size_t size) {
    return is_64_bit_ ? ScanStackForPointers<uint64_t>(data, size)
                      : ScanStackForPointers<uint32_t>(data, size);
  }

 private:
  template <typename Pointer>
  bool ScanStackForPointers(void* data, size_t size) {
    size_t sp_offset;
    if (!AssignIfInRange(&sp_offset, stack_pointer_ - stack_->Address())) {
      return false;
    }
    const size_t aligned_sp_offset =
        (sp_offset + sizeof(Pointer) - 1) & ~(sizeof(Pointer) - 1);

    auto words = reinterpret_cast<Pointer*>(static_cast<char*>(data) +
                                            aligned_sp_offset);
    size_t word_count = (size - aligned_sp_offset) / sizeof(Pointer);
    for (size_t index = 0; index < word_count; ++index) {
      if (words[index] >= low_ && words[index] < high_) {
        return true;
      }
    }

    return false;
  }

  VMAddress stack_pointer_;
  VMAddress low_;
  VMAddress high_;
  const MemorySnapshot* stack_;
  bool is_64_bit_;
};

}  // namespace

ProcessSnapshotSanitized::ProcessSnapshotSanitized() = default;

ProcessSnapshotSanitized::~ProcessSnapshotSanitized() = default;

bool ProcessSnapshotSanitized::Initialize(
    const ProcessSnapshot* snapshot,
    const std::vector<std::string>* annotations_whitelist,
    VMAddress target_module_address,
    bool sanitize_stacks) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  snapshot_ = snapshot;
  annotations_whitelist_ = annotations_whitelist;
  sanitize_stacks_ = sanitize_stacks;

  if (target_module_address) {
    const ExceptionSnapshot* exception = snapshot_->Exception();
    if (!exception) {
      return false;
    }

    const ThreadSnapshot* exc_thread = nullptr;
    for (const auto thread : snapshot_->Threads()) {
      if (thread->ThreadID() == exception->ThreadID()) {
        exc_thread = thread;
        break;
      }
    }
    if (!exc_thread) {
      return false;
    }

    const ModuleSnapshot* target_module = nullptr;
    for (const auto module : snapshot_->Modules()) {
      if (target_module_address >= module->Address() &&
          target_module_address < module->Address() + module->Size()) {
        target_module = module;
        break;
      }
    }
    if (!target_module) {
      return false;
    }

    // Only allow the snapshot if the program counter or some address on the
    // stack points into the target module.
    VMAddress pc = exception->Context()->InstructionPointer();
    VMAddress module_address_low = target_module->Address();
    VMAddress module_address_high = module_address_low + target_module->Size();
    if ((pc < module_address_low || pc >= module_address_high) &&
        !StackReferencesAddressRange().CheckStack(
            exc_thread->Stack(),
            exception->Context()->StackPointer(),
            module_address_low,
            module_address_high,
            exception->Context()->Is64Bit())) {
      return false;
    }
  }

  if (annotations_whitelist_) {
    for (const auto module : snapshot_->Modules()) {
      modules_.emplace_back(std::make_unique<internal::ModuleSnapshotSanitized>(
          module, annotations_whitelist_));
    }
  }

  if (sanitize_stacks_) {
    for (const auto module : snapshot_->Modules()) {
      address_ranges_.Insert(module->Address(), module->Size());
    }

    for (const auto thread : snapshot_->Threads()) {
      address_ranges_.Insert(thread->Stack()->Address(),
                             thread->Stack()->Size());
      threads_.emplace_back(std::make_unique<internal::ThreadSnapshotSanitized>(
          thread, &address_ranges_));
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

pid_t ProcessSnapshotSanitized::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->ProcessID();
}

pid_t ProcessSnapshotSanitized::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->ParentProcessID();
}

void ProcessSnapshotSanitized::SnapshotTime(timeval* snapshot_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  snapshot_->SnapshotTime(snapshot_time);
}

void ProcessSnapshotSanitized::ProcessStartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  snapshot_->ProcessStartTime(start_time);
}

void ProcessSnapshotSanitized::ProcessCPUTimes(timeval* user_time,
                                               timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  snapshot_->ProcessCPUTimes(user_time, system_time);
}

void ProcessSnapshotSanitized::ReportID(UUID* report_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  snapshot_->ReportID(report_id);
}

void ProcessSnapshotSanitized::ClientID(UUID* client_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  snapshot_->ClientID(client_id);
}

const std::map<std::string, std::string>&
ProcessSnapshotSanitized::AnnotationsSimpleMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->AnnotationsSimpleMap();
}

const SystemSnapshot* ProcessSnapshotSanitized::System() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->System();
}

std::vector<const ThreadSnapshot*> ProcessSnapshotSanitized::Threads() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!sanitize_stacks_) {
    return snapshot_->Threads();
  }

  std::vector<const ThreadSnapshot*> threads;
  for (const auto& thread : threads_) {
    threads.push_back(thread.get());
  }
  return threads;
}

std::vector<const ModuleSnapshot*> ProcessSnapshotSanitized::Modules() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!annotations_whitelist_) {
    return snapshot_->Modules();
  }

  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> ProcessSnapshotSanitized::UnloadedModules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->UnloadedModules();
}

const ExceptionSnapshot* ProcessSnapshotSanitized::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->Exception();
}

std::vector<const MemoryMapRegionSnapshot*>
ProcessSnapshotSanitized::MemoryMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->MemoryMap();
}

std::vector<HandleSnapshot> ProcessSnapshotSanitized::Handles() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->Handles();
}

std::vector<const MemorySnapshot*> ProcessSnapshotSanitized::ExtraMemory()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return snapshot_->ExtraMemory();
}

}  // namespace crashpad
