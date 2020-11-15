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

#include "snapshot/win/process_snapshot_win.h"

#include <stddef.h>
#include <wchar.h>

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "util/misc/from_pointer_cast.h"
#include "util/misc/time.h"
#include "util/win/nt_internals.h"
#include "util/win/registration_protocol_win.h"

namespace crashpad {

ProcessSnapshotWin::ProcessSnapshotWin()
    : ProcessSnapshot(),
      system_(),
      threads_(),
      modules_(),
      exception_(),
      memory_map_(),
      process_reader_(),
      report_id_(),
      client_id_(),
      annotations_simple_map_(),
      snapshot_time_(),
      options_(),
      initialized_() {
}

ProcessSnapshotWin::~ProcessSnapshotWin() {
}

bool ProcessSnapshotWin::Initialize(
    HANDLE process,
    ProcessSuspensionState suspension_state,
    WinVMAddress exception_information_address,
    WinVMAddress debug_critical_section_address) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  GetTimeOfDay(&snapshot_time_);

  if (!process_reader_.Initialize(process, suspension_state))
    return false;

  if (exception_information_address != 0) {
    ExceptionInformation exception_information = {};
    if (!process_reader_.ReadMemory(exception_information_address,
                                    sizeof(exception_information),
                                    &exception_information)) {
      LOG(WARNING) << "ReadMemory ExceptionInformation failed";
      return false;
    }

    exception_.reset(new internal::ExceptionSnapshotWin());
    if (!exception_->Initialize(&process_reader_,
                                exception_information.thread_id,
                                exception_information.exception_pointers)) {
      exception_.reset();
      return false;
    }
  }


  system_.Initialize(&process_reader_);

  if (process_reader_.Is64Bit()) {
    InitializePebData<process_types::internal::Traits64>(
        debug_critical_section_address);
  } else {
    InitializePebData<process_types::internal::Traits32>(
        debug_critical_section_address);
  }

  InitializeModules();
  InitializeUnloadedModules();

  GetCrashpadOptionsInternal(&options_);

  InitializeThreads(
      options_.gather_indirectly_referenced_memory == TriState::kEnabled,
      options_.indirectly_referenced_memory_cap);

  for (const MEMORY_BASIC_INFORMATION64& mbi :
       process_reader_.GetProcessInfo().MemoryInfo()) {
    memory_map_.push_back(
        std::make_unique<internal::MemoryMapRegionSnapshotWin>(mbi));
  }

  for (const auto& module : modules_) {
    for (const auto& range : module->ExtraMemoryRanges()) {
      AddMemorySnapshot(range.base(), range.size(), &extra_memory_);
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

void ProcessSnapshotWin::GetCrashpadOptions(
    CrashpadInfoClientOptions* options) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *options = options_;
}

pid_t ProcessSnapshotWin::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_reader_.GetProcessInfo().ProcessID();
}

pid_t ProcessSnapshotWin::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_reader_.GetProcessInfo().ParentProcessID();
}

void ProcessSnapshotWin::SnapshotTime(timeval* snapshot_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *snapshot_time = snapshot_time_;
}

void ProcessSnapshotWin::ProcessStartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  process_reader_.StartTime(start_time);
}

void ProcessSnapshotWin::ProcessCPUTimes(timeval* user_time,
                                         timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  process_reader_.CPUTimes(user_time, system_time);
}

void ProcessSnapshotWin::ReportID(UUID* report_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *report_id = report_id_;
}

void ProcessSnapshotWin::ClientID(UUID* client_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *client_id = client_id_;
}

const std::map<std::string, std::string>&
ProcessSnapshotWin::AnnotationsSimpleMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_simple_map_;
}

const SystemSnapshot* ProcessSnapshotWin::System() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &system_;
}

std::vector<const ThreadSnapshot*> ProcessSnapshotWin::Threads() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ThreadSnapshot*> threads;
  for (const auto& thread : threads_) {
    threads.push_back(thread.get());
  }
  return threads;
}

std::vector<const ModuleSnapshot*> ProcessSnapshotWin::Modules() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> ProcessSnapshotWin::UnloadedModules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return unloaded_modules_;
}

const ExceptionSnapshot* ProcessSnapshotWin::Exception() const {
  return exception_.get();
}

std::vector<const MemoryMapRegionSnapshot*> ProcessSnapshotWin::MemoryMap()
    const {
  std::vector<const MemoryMapRegionSnapshot*> memory_map;
  for (const auto& item : memory_map_) {
    memory_map.push_back(item.get());
  }
  return memory_map;
}

std::vector<HandleSnapshot> ProcessSnapshotWin::Handles() const {
  std::vector<HandleSnapshot> result;
  for (const auto& handle : process_reader_.GetProcessInfo().Handles()) {
    HandleSnapshot snapshot;
    // This is probably not strictly correct, but these are not localized so we
    // expect them all to be in ASCII range anyway. This will need to be more
    // carefully done if the object name is added.
    snapshot.type_name = base::UTF16ToUTF8(handle.type_name);
    snapshot.handle = handle.handle;
    snapshot.attributes = handle.attributes;
    snapshot.granted_access = handle.granted_access;
    snapshot.pointer_count = handle.pointer_count;
    snapshot.handle_count = handle.handle_count;
    result.push_back(snapshot);
  }
  return result;
}

std::vector<const MemorySnapshot*> ProcessSnapshotWin::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const MemorySnapshot*> extra_memory;
  for (const auto& em : extra_memory_) {
    extra_memory.push_back(em.get());
  }
  return extra_memory;
}

void ProcessSnapshotWin::InitializeThreads(
    bool gather_indirectly_referenced_memory,
    uint32_t indirectly_referenced_memory_cap) {
  const std::vector<ProcessReaderWin::Thread>& process_reader_threads =
      process_reader_.Threads();
  uint32_t* budget_remaining_pointer = nullptr;
  uint32_t budget_remaining = indirectly_referenced_memory_cap;
  if (gather_indirectly_referenced_memory)
    budget_remaining_pointer = &budget_remaining;
  for (const ProcessReaderWin::Thread& process_reader_thread :
       process_reader_threads) {
    auto thread = std::make_unique<internal::ThreadSnapshotWin>();
    if (thread->Initialize(&process_reader_,
                           process_reader_thread,
                           budget_remaining_pointer)) {
      threads_.push_back(std::move(thread));
    }
  }
}

void ProcessSnapshotWin::InitializeModules() {
  const std::vector<ProcessInfo::Module>& process_reader_modules =
      process_reader_.Modules();
  for (const ProcessInfo::Module& process_reader_module :
       process_reader_modules) {
    auto module = std::make_unique<internal::ModuleSnapshotWin>();
    if (module->Initialize(&process_reader_, process_reader_module)) {
      modules_.push_back(std::move(module));
    }
  }
}

void ProcessSnapshotWin::InitializeUnloadedModules() {
  // As documented by https://msdn.microsoft.com/library/cc678403.aspx, we can
  // retrieve the location for our unload events, and use that address in the
  // target process. Unfortunately, this of course only works for 64-reading-64
  // and 32-reading-32, so at the moment, we simply do not retrieve unloaded
  // modules for 64-reading-32. See https://crashpad.chromium.org/bug/89.

#if defined(ARCH_CPU_X86_64)
  if (!process_reader_.Is64Bit()) {
    LOG(ERROR)
        << "reading unloaded modules across bitness not currently supported";
    return;
  }
  using Traits = process_types::internal::Traits64;
#elif defined(ARCH_CPU_X86)
  using Traits = process_types::internal::Traits32;
#else
#error port
#endif

  ULONG* element_size;
  ULONG* element_count;
  void* event_trace_address;
  RtlGetUnloadEventTraceEx(&element_size, &element_count, &event_trace_address);

  if (*element_size < sizeof(RTL_UNLOAD_EVENT_TRACE<Traits>)) {
    LOG(ERROR) << "unexpected unloaded module list element size";
    return;
  }

  const WinVMAddress address_in_target_process =
      FromPointerCast<WinVMAddress>(event_trace_address);

  Traits::Pointer pointer_to_array;
  if (!process_reader_.ReadMemory(address_in_target_process,
                                  sizeof(pointer_to_array),
                                  &pointer_to_array)) {
    LOG(ERROR) << "failed to read target address";
    return;
  }

  // No unloaded modules.
  if (pointer_to_array == 0)
    return;

  const size_t data_size = *element_size * *element_count;
  std::vector<uint8_t> data(data_size);
  if (!process_reader_.ReadMemory(pointer_to_array, data_size, &data[0])) {
    LOG(ERROR) << "failed to read unloaded module data";
    return;
  }

  for (ULONG i = 0; i < *element_count; ++i) {
    const uint8_t* base_address = &data[i * *element_size];
    const auto& uet =
        *reinterpret_cast<const RTL_UNLOAD_EVENT_TRACE<Traits>*>(base_address);
    if (uet.ImageName[0] != 0) {
      unloaded_modules_.push_back(UnloadedModuleSnapshot(
          uet.BaseAddress,
          uet.SizeOfImage,
          uet.CheckSum,
          uet.TimeDateStamp,
          base::UTF16ToUTF8(base::StringPiece16(
              uet.ImageName,
              wcsnlen(uet.ImageName, arraysize(uet.ImageName))))));
    }
  }
}

void ProcessSnapshotWin::GetCrashpadOptionsInternal(
    CrashpadInfoClientOptions* options) {
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

template <class Traits>
void ProcessSnapshotWin::InitializePebData(
    WinVMAddress debug_critical_section_address) {
  WinVMAddress peb_address;
  WinVMSize peb_size;
  process_reader_.GetProcessInfo().Peb(&peb_address, &peb_size);
  AddMemorySnapshot(peb_address, peb_size, &extra_memory_);

  process_types::PEB<Traits> peb_data;
  if (!process_reader_.ReadMemory(peb_address, peb_size, &peb_data)) {
    LOG(ERROR) << "ReadMemory PEB";
    return;
  }

  process_types::PEB_LDR_DATA<Traits> peb_ldr_data;
  AddMemorySnapshot(peb_data.Ldr, sizeof(peb_ldr_data), &extra_memory_);
  if (!process_reader_.ReadMemory(
          peb_data.Ldr, sizeof(peb_ldr_data), &peb_ldr_data)) {
    LOG(ERROR) << "ReadMemory PEB_LDR_DATA";
  } else {
    // Walk the LDR structure to retrieve its pointed-to data.
    AddMemorySnapshotForLdrLIST_ENTRY(
        peb_ldr_data.InLoadOrderModuleList,
        offsetof(process_types::LDR_DATA_TABLE_ENTRY<Traits>, InLoadOrderLinks),
        &extra_memory_);
    AddMemorySnapshotForLdrLIST_ENTRY(
        peb_ldr_data.InMemoryOrderModuleList,
        offsetof(process_types::LDR_DATA_TABLE_ENTRY<Traits>,
                 InMemoryOrderLinks),
        &extra_memory_);
    AddMemorySnapshotForLdrLIST_ENTRY(
        peb_ldr_data.InInitializationOrderModuleList,
        offsetof(process_types::LDR_DATA_TABLE_ENTRY<Traits>,
                 InInitializationOrderLinks),
        &extra_memory_);
  }

  process_types::RTL_USER_PROCESS_PARAMETERS<Traits> process_parameters;
  if (!process_reader_.ReadMemory(peb_data.ProcessParameters,
                                  sizeof(process_parameters),
                                  &process_parameters)) {
    LOG(ERROR) << "ReadMemory RTL_USER_PROCESS_PARAMETERS";
    return;
  }
  AddMemorySnapshot(
      peb_data.ProcessParameters, sizeof(process_parameters), &extra_memory_);

  AddMemorySnapshotForUNICODE_STRING(
      process_parameters.CurrentDirectory.DosPath, &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.DllPath,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.ImagePathName,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.CommandLine,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.WindowTitle,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.DesktopInfo,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.ShellInfo,
                                     &extra_memory_);
  AddMemorySnapshotForUNICODE_STRING(process_parameters.RuntimeData,
                                     &extra_memory_);
  AddMemorySnapshot(
      process_parameters.Environment,
      DetermineSizeOfEnvironmentBlock(process_parameters.Environment),
      &extra_memory_);

  // Walk the loader lock which is directly referenced by the PEB.
  ReadLock<Traits>(peb_data.LoaderLock, &extra_memory_);

  // TODO(scottmg): Use debug_critical_section_address to walk the list of
  // locks (see history of this file for walking code). In some configurations
  // this can walk many thousands of locks, so we may want to get some
  // annotation from the client for which locks to grab. Unfortunately, without
  // walking the list, the !locks command in windbg won't work because it
  // requires the lock pointed to by ntdll!RtlCriticalSectionList, which we
  // won't have captured.
}

void ProcessSnapshotWin::AddMemorySnapshot(
    WinVMAddress address,
    WinVMSize size,
    std::vector<std::unique_ptr<internal::MemorySnapshotWin>>* into) {
  if (size == 0)
    return;

  if (!process_reader_.GetProcessInfo().LoggingRangeIsFullyReadable(
          CheckedRange<WinVMAddress, WinVMSize>(address, size))) {
    return;
  }

  // If we have already added this exact range, don't add it again. This is
  // useful for the LDR module lists which are a set of doubly-linked lists, all
  // pointing to the same module name strings.
  // TODO(scottmg): A more general version of this, handling overlapping,
  // contained, etc. https://crashpad.chromium.org/bug/61.
  for (const auto& memory_snapshot : *into) {
    if (memory_snapshot->Address() == address &&
        memory_snapshot->Size() == size) {
      return;
    }
  }

  into->push_back(std::make_unique<internal::MemorySnapshotWin>());
  into->back()->Initialize(&process_reader_, address, size);
}

template <class Traits>
void ProcessSnapshotWin::AddMemorySnapshotForUNICODE_STRING(
    const process_types::UNICODE_STRING<Traits>& us,
    std::vector<std::unique_ptr<internal::MemorySnapshotWin>>* into) {
  AddMemorySnapshot(us.Buffer, us.Length, into);
}

template <class Traits>
void ProcessSnapshotWin::AddMemorySnapshotForLdrLIST_ENTRY(
    const process_types::LIST_ENTRY<Traits>& le,
    size_t offset_of_member,
    std::vector<std::unique_ptr<internal::MemorySnapshotWin>>* into) {
  // Walk the doubly-linked list of entries, adding the list memory itself, as
  // well as pointed-to strings.
  typename Traits::Pointer last = le.Blink;
  process_types::LDR_DATA_TABLE_ENTRY<Traits> entry;
  typename Traits::Pointer cur = le.Flink;
  for (;;) {
    // |cur| is the pointer to LIST_ENTRY embedded in the LDR_DATA_TABLE_ENTRY.
    // So we need to offset back to the beginning of the structure.
    if (!process_reader_.ReadMemory(
            cur - offset_of_member, sizeof(entry), &entry)) {
      return;
    }
    AddMemorySnapshot(cur - offset_of_member, sizeof(entry), into);
    AddMemorySnapshotForUNICODE_STRING(entry.FullDllName, into);
    AddMemorySnapshotForUNICODE_STRING(entry.BaseDllName, into);

    process_types::LIST_ENTRY<Traits>* links =
        reinterpret_cast<process_types::LIST_ENTRY<Traits>*>(
            reinterpret_cast<unsigned char*>(&entry) + offset_of_member);
    cur = links->Flink;
    if (cur == last)
      break;
  }
}

WinVMSize ProcessSnapshotWin::DetermineSizeOfEnvironmentBlock(
    WinVMAddress start_of_environment_block) {
  // https://blogs.msdn.microsoft.com/oldnewthing/20100203-00/?p=15083: On newer
  // OSs there's no stated limit, but in practice grabbing 32k characters should
  // be more than enough.
  std::wstring env_block;
  env_block.resize(32768);
  WinVMSize bytes_read = process_reader_.ReadAvailableMemory(
      start_of_environment_block,
      env_block.size() * sizeof(env_block[0]),
      &env_block[0]);
  env_block.resize(
      static_cast<unsigned int>(bytes_read / sizeof(env_block[0])));
  static constexpr wchar_t terminator[] = {0, 0};
  size_t at = env_block.find(std::wstring(terminator, arraysize(terminator)));
  if (at != std::wstring::npos)
    env_block.resize(at + arraysize(terminator));

  return env_block.size() * sizeof(env_block[0]);
}

template <class Traits>
void ProcessSnapshotWin::ReadLock(
    WinVMAddress start,
    std::vector<std::unique_ptr<internal::MemorySnapshotWin>>* into) {
  // We're walking the RTL_CRITICAL_SECTION_DEBUG ProcessLocksList, but starting
  // from an actual RTL_CRITICAL_SECTION, so start by getting to the first
  // RTL_CRITICAL_SECTION_DEBUG.

  process_types::RTL_CRITICAL_SECTION<Traits> critical_section;
  if (!process_reader_.ReadMemory(
          start, sizeof(critical_section), &critical_section)) {
    LOG(ERROR) << "failed to read RTL_CRITICAL_SECTION";
    return;
  }

  AddMemorySnapshot(
      start, sizeof(process_types::RTL_CRITICAL_SECTION<Traits>), into);

  constexpr decltype(critical_section.DebugInfo) kInvalid =
      static_cast<decltype(critical_section.DebugInfo)>(-1);
  if (critical_section.DebugInfo == kInvalid)
    return;

  AddMemorySnapshot(critical_section.DebugInfo,
                    sizeof(process_types::RTL_CRITICAL_SECTION_DEBUG<Traits>),
                    into);
}

}  // namespace crashpad
