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

#include "snapshot/linux/process_reader_linux.h"

#include <elf.h>
#include <errno.h>
#include <sched.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>

#include "base/logging.h"
#include "build/build_config.h"
#include "snapshot/linux/debug_rendezvous.h"
#include "util/linux/auxiliary_vector.h"
#include "util/linux/proc_stat_reader.h"

namespace crashpad {

namespace {

bool ShouldMergeStackMappings(const MemoryMap::Mapping& stack_mapping,
                              const MemoryMap::Mapping& adj_mapping) {
  DCHECK(stack_mapping.readable);
  return adj_mapping.readable && stack_mapping.device == adj_mapping.device &&
         stack_mapping.inode == adj_mapping.inode &&
         (stack_mapping.name == adj_mapping.name ||
          stack_mapping.name.empty() || adj_mapping.name.empty());
}

}  // namespace

ProcessReaderLinux::Thread::Thread()
    : thread_info(),
      stack_region_address(0),
      stack_region_size(0),
      tid(-1),
      static_priority(-1),
      nice_value(-1) {}

ProcessReaderLinux::Thread::~Thread() {}

bool ProcessReaderLinux::Thread::InitializePtrace(
    PtraceConnection* connection) {
  if (!connection->GetThreadInfo(tid, &thread_info)) {
    return false;
  }

  // TODO(jperaza): Collect scheduling priorities via the broker when they can't
  // be collected directly.
  have_priorities = false;

  // TODO(jperaza): Starting with Linux 3.14, scheduling policy, static
  // priority, and nice value can be collected all in one call with
  // sched_getattr().
  int res = sched_getscheduler(tid);
  if (res < 0) {
    PLOG(WARNING) << "sched_getscheduler";
    return true;
  }
  sched_policy = res;

  sched_param param;
  if (sched_getparam(tid, &param) != 0) {
    PLOG(WARNING) << "sched_getparam";
    return true;
  }
  static_priority = param.sched_priority;

  errno = 0;
  res = getpriority(PRIO_PROCESS, tid);
  if (res == -1 && errno) {
    PLOG(WARNING) << "getpriority";
    return true;
  }
  nice_value = res;

  have_priorities = true;
  return true;
}

void ProcessReaderLinux::Thread::InitializeStack(ProcessReaderLinux* reader) {
  LinuxVMAddress stack_pointer;
#if defined(ARCH_CPU_X86_FAMILY)
  stack_pointer = reader->Is64Bit() ? thread_info.thread_context.t64.rsp
                                    : thread_info.thread_context.t32.esp;
#elif defined(ARCH_CPU_ARM_FAMILY)
  stack_pointer = reader->Is64Bit() ? thread_info.thread_context.t64.sp
                                    : thread_info.thread_context.t32.sp;
#elif defined(ARCH_CPU_MIPS_FAMILY)
  stack_pointer = reader->Is64Bit() ? thread_info.thread_context.t64.regs[29]
                                    : thread_info.thread_context.t32.regs[29];
#else
#error Port.
#endif
  InitializeStackFromSP(reader, stack_pointer);
}

void ProcessReaderLinux::Thread::InitializeStackFromSP(
    ProcessReaderLinux* reader,
    LinuxVMAddress stack_pointer) {
  const MemoryMap* memory_map = reader->GetMemoryMap();

  // If we can't find the mapping, it's probably a bad stack pointer
  const MemoryMap::Mapping* mapping = memory_map->FindMapping(stack_pointer);
  if (!mapping) {
    LOG(WARNING) << "no stack mapping";
    return;
  }
  LinuxVMAddress stack_region_start = stack_pointer;

  // We've hit what looks like a guard page; skip to the end and check for a
  // mapped stack region.
  if (!mapping->readable) {
    stack_region_start = mapping->range.End();
    mapping = memory_map->FindMapping(stack_region_start);
    if (!mapping) {
      LOG(WARNING) << "no stack mapping";
      return;
    }
  } else {
#if defined(ARCH_CPU_X86_FAMILY)
    // Adjust start address to include the red zone
    if (reader->Is64Bit()) {
      constexpr LinuxVMSize kRedZoneSize = 128;
      LinuxVMAddress red_zone_base =
          stack_region_start - std::min(kRedZoneSize, stack_region_start);

      // Only include the red zone if it is part of a valid mapping
      if (red_zone_base >= mapping->range.Base()) {
        stack_region_start = red_zone_base;
      } else {
        const MemoryMap::Mapping* rz_mapping =
            memory_map->FindMapping(red_zone_base);
        if (rz_mapping && ShouldMergeStackMappings(*mapping, *rz_mapping)) {
          stack_region_start = red_zone_base;
        } else {
          stack_region_start = mapping->range.Base();
        }
      }
    }
#endif
  }
  stack_region_address = stack_region_start;

  // If there are more mappings at the end of this one, they may be a
  // continuation of the stack.
  LinuxVMAddress stack_end = mapping->range.End();
  const MemoryMap::Mapping* next_mapping;
  while ((next_mapping = memory_map->FindMapping(stack_end)) &&
         ShouldMergeStackMappings(*mapping, *next_mapping)) {
    stack_end = next_mapping->range.End();
  }

  // The main thread should have an entry in the maps file just for its stack,
  // so we'll assume the base of the stack is at the end of the region. Other
  // threads' stacks may not have their own entries in the maps file if they
  // were user-allocated within a larger mapping, but pthreads places the TLS
  // at the high-address end of the stack so we can try using that to shrink
  // the stack region.
  stack_region_size = stack_end - stack_region_address;
  if (tid != reader->ProcessID() &&
      thread_info.thread_specific_data_address > stack_region_address &&
      thread_info.thread_specific_data_address < stack_end) {
    stack_region_size =
        thread_info.thread_specific_data_address - stack_region_address;
  }
}

ProcessReaderLinux::Module::Module()
    : name(), elf_reader(nullptr), type(ModuleSnapshot::kModuleTypeUnknown) {}

ProcessReaderLinux::Module::~Module() = default;

ProcessReaderLinux::ProcessReaderLinux()
    : connection_(),
      process_info_(),
      memory_map_(),
      threads_(),
      modules_(),
      elf_readers_(),
      is_64_bit_(false),
      initialized_threads_(false),
      initialized_modules_(false),
      initialized_() {}

ProcessReaderLinux::~ProcessReaderLinux() {}

bool ProcessReaderLinux::Initialize(PtraceConnection* connection) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  DCHECK(connection);
  connection_ = connection;

  if (!process_info_.InitializeWithPtrace(connection_)) {
    return false;
  }

  if (!memory_map_.Initialize(connection_)) {
    return false;
  }

  is_64_bit_ = process_info_.Is64Bit();

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessReaderLinux::StartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_info_.StartTime(start_time);
}

bool ProcessReaderLinux::CPUTimes(timeval* user_time,
                                  timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  timerclear(user_time);
  timerclear(system_time);

  timeval local_user_time;
  timerclear(&local_user_time);
  timeval local_system_time;
  timerclear(&local_system_time);

  for (const Thread& thread : threads_) {
    ProcStatReader stat;
    if (!stat.Initialize(connection_, thread.tid)) {
      return false;
    }

    timeval thread_user_time;
    if (!stat.UserCPUTime(&thread_user_time)) {
      return false;
    }

    timeval thread_system_time;
    if (!stat.SystemCPUTime(&thread_system_time)) {
      return false;
    }

    timeradd(&local_user_time, &thread_user_time, &local_user_time);
    timeradd(&local_system_time, &thread_system_time, &local_system_time);
  }

  *user_time = local_user_time;
  *system_time = local_system_time;
  return true;
}

const std::vector<ProcessReaderLinux::Thread>& ProcessReaderLinux::Threads() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!initialized_threads_) {
    InitializeThreads();
  }
  return threads_;
}

const std::vector<ProcessReaderLinux::Module>& ProcessReaderLinux::Modules() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!initialized_modules_) {
    InitializeModules();
  }
  return modules_;
}

void ProcessReaderLinux::InitializeThreads() {
  DCHECK(threads_.empty());
  initialized_threads_ = true;

  pid_t pid = ProcessID();
  if (pid == getpid()) {
    // TODO(jperaza): ptrace can't be used on threads in the same thread group.
    // Using clone to create a new thread in it's own thread group doesn't work
    // because glibc doesn't support threads it didn't create via pthreads.
    // Fork a new process to snapshot us and copy the data back?
    LOG(ERROR) << "not implemented";
    return;
  }

  Thread main_thread;
  main_thread.tid = pid;
  if (main_thread.InitializePtrace(connection_)) {
    main_thread.InitializeStack(this);
    threads_.push_back(main_thread);
  } else {
    LOG(WARNING) << "Couldn't initialize main thread.";
  }

  bool main_thread_found = false;
  std::vector<pid_t> thread_ids;
  bool result = connection_->Threads(&thread_ids);
  DCHECK(result);
  for (pid_t tid : thread_ids) {
    if (tid == pid) {
      DCHECK(!main_thread_found);
      main_thread_found = true;
      continue;
    }

    Thread thread;
    thread.tid = tid;
    if (connection_->Attach(tid) && thread.InitializePtrace(connection_)) {
      thread.InitializeStack(this);
      threads_.push_back(thread);
    }
  }
  DCHECK(main_thread_found);
}

void ProcessReaderLinux::InitializeModules() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  initialized_modules_ = true;

  AuxiliaryVector aux;
  if (!aux.Initialize(connection_)) {
    return;
  }

  LinuxVMAddress phdrs;
  if (!aux.GetValue(AT_PHDR, &phdrs)) {
    return;
  }

  ProcessMemoryRange range;
  if (!range.Initialize(Memory(), is_64_bit_)) {
    return;
  }

  // The strategy used for identifying loaded modules depends on ELF files
  // conventionally loading their header and program headers into memory.
  // Locating the correct module could fail if the headers aren't mapped, are
  // mapped at an unexpected location, or if there are other mappings
  // constructed to look like the ELF module being searched for.
  const MemoryMap::Mapping* exe_mapping = nullptr;
  std::unique_ptr<ElfImageReader> exe_reader;
  {
    const MemoryMap::Mapping* phdr_mapping = memory_map_.FindMapping(phdrs);
    if (!phdr_mapping) {
      return;
    }

    std::vector<const MemoryMap::Mapping*> possible_mappings =
        memory_map_.FindFilePossibleMmapStarts(*phdr_mapping);
    for (auto riter = possible_mappings.rbegin();
         riter != possible_mappings.rend();
         ++riter) {
      auto mapping = *riter;
      auto parsed_exe = std::make_unique<ElfImageReader>();
      if (parsed_exe->Initialize(
              range,
              mapping->range.Base(),
              /* verbose= */ possible_mappings.size() == 1) &&
          parsed_exe->GetProgramHeaderTableAddress() == phdrs) {
        exe_mapping = mapping;
        exe_reader = std::move(parsed_exe);
        break;
      }
    }
    if (!exe_mapping) {
      LOG(ERROR) << "no exe mappings " << possible_mappings.size();
      return;
    }
  }

  LinuxVMAddress debug_address;
  if (!exe_reader->GetDebugAddress(&debug_address)) {
    return;
  }

  DebugRendezvous debug;
  if (!debug.Initialize(range, debug_address)) {
    return;
  }

  Module exe = {};
  exe.name = !debug.Executable()->name.empty() ? debug.Executable()->name
                                               : exe_mapping->name;
  exe.elf_reader = exe_reader.get();
  exe.type = ModuleSnapshot::ModuleType::kModuleTypeExecutable;

  modules_.push_back(exe);
  elf_readers_.push_back(std::move(exe_reader));

  LinuxVMAddress loader_base = 0;
  aux.GetValue(AT_BASE, &loader_base);

  for (const DebugRendezvous::LinkEntry& entry : debug.Modules()) {
    const MemoryMap::Mapping* module_mapping = nullptr;
    std::unique_ptr<ElfImageReader> elf_reader;
    {
      const MemoryMap::Mapping* dyn_mapping =
          memory_map_.FindMapping(entry.dynamic_array);
      if (!dyn_mapping) {
        continue;
      }

      std::vector<const MemoryMap::Mapping*> possible_mappings =
          memory_map_.FindFilePossibleMmapStarts(*dyn_mapping);
      for (auto riter = possible_mappings.rbegin();
           riter != possible_mappings.rend();
           ++riter) {
        auto mapping = *riter;
        auto parsed_module = std::make_unique<ElfImageReader>();
        VMAddress dynamic_address;
        if (parsed_module->Initialize(
                range,
                mapping->range.Base(),
                /* verbose= */ possible_mappings.size() == 1) &&
            parsed_module->GetDynamicArrayAddress(&dynamic_address) &&
            dynamic_address == entry.dynamic_array) {
          module_mapping = mapping;
          elf_reader = std::move(parsed_module);
          break;
        }
      }
      if (!module_mapping) {
        LOG(ERROR) << "no module mappings " << possible_mappings.size();
        continue;
      }
    }

    Module module = {};
    std::string soname;
    if (elf_reader->SoName(&soname) && !soname.empty()) {
      module.name = soname;
    } else {
      module.name = !entry.name.empty() ? entry.name : module_mapping->name;
    }
    module.elf_reader = elf_reader.get();
    module.type = loader_base && elf_reader->Address() == loader_base
                      ? ModuleSnapshot::kModuleTypeDynamicLoader
                      : ModuleSnapshot::kModuleTypeSharedLibrary;
    modules_.push_back(module);
    elf_readers_.push_back(std::move(elf_reader));
  }
}

}  // namespace crashpad
