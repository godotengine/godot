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

#include "snapshot/fuchsia/process_reader_fuchsia.h"

#include <lib/zx/thread.h>
#include <link.h>
#include <zircon/syscalls.h>

#include "base/fuchsia/fuchsia_logging.h"
#include "base/logging.h"
#include "util/fuchsia/koid_utilities.h"

namespace crashpad {

namespace {

// Based on the thread's SP and the process's memory map, attempts to figure out
// the stack regions for the thread. Fuchsia's C ABI specifies
// https://fuchsia.googlesource.com/zircon/+/master/docs/safestack.md so the
// callstack and locals-that-have-their-address-taken are in two different
// stacks.
void GetStackRegions(
    const zx_thread_state_general_regs_t& regs,
    const MemoryMapFuchsia& memory_map,
    std::vector<CheckedRange<zx_vaddr_t, size_t>>* stack_regions) {
  stack_regions->clear();

  uint64_t sp;
#if defined(ARCH_CPU_X86_64)
  sp = regs.rsp;
#elif defined(ARCH_CPU_ARM64)
  sp = regs.sp;
#else
#error Port
#endif

  zx_info_maps_t range_with_sp;
  if (!memory_map.FindMappingForAddress(sp, &range_with_sp)) {
    LOG(ERROR) << "stack pointer not found in mapping";
    return;
  }

  if (range_with_sp.type != ZX_INFO_MAPS_TYPE_MAPPING) {
    LOG(ERROR) << "stack range has unexpected type, continuing anyway";
  }

  if (range_with_sp.u.mapping.mmu_flags & ZX_VM_FLAG_PERM_EXECUTE) {
    LOG(ERROR)
        << "stack range is unexpectedly marked executable, continuing anyway";
  }

  stack_regions->push_back(
      CheckedRange<zx_vaddr_t, size_t>(range_with_sp.base, range_with_sp.size));

  // TODO(scottmg): https://crashpad.chromium.org/bug/196, once the retrievable
  // registers include FS and similar for ARM, retrieve the region for the
  // unsafe part of the stack too.
}

}  // namespace

ProcessReaderFuchsia::Module::Module() = default;

ProcessReaderFuchsia::Module::~Module() = default;

ProcessReaderFuchsia::Thread::Thread() = default;

ProcessReaderFuchsia::Thread::~Thread() = default;

ProcessReaderFuchsia::ProcessReaderFuchsia() = default;

ProcessReaderFuchsia::~ProcessReaderFuchsia() = default;

bool ProcessReaderFuchsia::Initialize(const zx::process& process) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_ = zx::unowned_process(process);

  process_memory_.reset(new ProcessMemoryFuchsia());
  process_memory_->Initialize(*process_);

  memory_map_.Initialize(*process_);

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const std::vector<ProcessReaderFuchsia::Module>&
ProcessReaderFuchsia::Modules() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!initialized_modules_) {
    InitializeModules();
  }

  return modules_;
}

const std::vector<ProcessReaderFuchsia::Thread>&
ProcessReaderFuchsia::Threads() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!initialized_threads_) {
    InitializeThreads();
  }

  return threads_;
}

void ProcessReaderFuchsia::InitializeModules() {
  DCHECK(!initialized_modules_);
  DCHECK(modules_.empty());

  initialized_modules_ = true;

  // TODO(scottmg): <inspector/inspector.h> does some of this, but doesn't
  // expose any of the data that's necessary to fill out a Module after it
  // retrieves (some of) the data into internal structures. It may be worth
  // trying to refactor/upstream some of this into Fuchsia.

  std::string app_name;
  {
    char name[ZX_MAX_NAME_LEN];
    zx_status_t status =
        process_->get_property(ZX_PROP_NAME, name, sizeof(name));
    if (status != ZX_OK) {
      LOG(ERROR) << "zx_object_get_property ZX_PROP_NAME";
      return;
    }

    app_name = name;
  }

  // Starting from the ld.so's _dl_debug_addr, read the link_map structure and
  // walk the list to fill out modules_.

  uintptr_t debug_address;
  zx_status_t status = process_->get_property(
      ZX_PROP_PROCESS_DEBUG_ADDR, &debug_address, sizeof(debug_address));
  if (status != ZX_OK || debug_address == 0) {
    LOG(ERROR) << "zx_object_get_property ZX_PROP_PROCESS_DEBUG_ADDR";
    return;
  }

  constexpr auto k_r_debug_map_offset = offsetof(r_debug, r_map);
  uintptr_t map;
  if (!process_memory_->Read(
          debug_address + k_r_debug_map_offset, sizeof(map), &map)) {
    LOG(ERROR) << "read link_map";
    return;
  }

  int i = 0;
  constexpr int kMaxDso = 1000;  // Stop after an unreasonably large number.
  while (map != 0) {
    if (++i >= kMaxDso) {
      LOG(ERROR) << "possibly circular dso list, terminating";
      return;
    }

    constexpr auto k_link_map_addr_offset = offsetof(link_map, l_addr);
    zx_vaddr_t base;
    if (!process_memory_->Read(
            map + k_link_map_addr_offset, sizeof(base), &base)) {
      LOG(ERROR) << "Read base";
      // Could theoretically continue here, but realistically if any part of
      // link_map fails to read, things are looking bad, so just abort.
      break;
    }

    constexpr auto k_link_map_next_offset = offsetof(link_map, l_next);
    zx_vaddr_t next;
    if (!process_memory_->Read(
            map + k_link_map_next_offset, sizeof(next), &next)) {
      LOG(ERROR) << "Read next";
      break;
    }

    constexpr auto k_link_map_name_offset = offsetof(link_map, l_name);
    zx_vaddr_t name_address;
    if (!process_memory_->Read(map + k_link_map_name_offset,
                               sizeof(name_address),
                               &name_address)) {
      LOG(ERROR) << "Read name address";
      break;
    }

    std::string dsoname;
    if (!process_memory_->ReadCString(name_address, &dsoname)) {
      // In this case, it could be reasonable to continue on to the next module
      // as this data isn't strictly in the link_map.
      LOG(ERROR) << "ReadCString name";
    }

    // The vDSO is libzircon.so, but it's not actually loaded normally, it's
    // injected by the kernel, so doesn't have a normal name. When dump_syms is
    // run on libzircon.so, it uses that file name, and in order for the crash
    // server to match symbols both the debug id and the name of the binary have
    // to match. So, map from "<vDSO>" to "libzircon.so" so that symbol
    // resolution works correctly.
    if (dsoname == "<vDSO>") {
      dsoname = "libzircon.so";
    }

    Module module;
    if (dsoname.empty()) {
      module.name = app_name;
      module.type = ModuleSnapshot::kModuleTypeExecutable;
    } else {
      module.name = dsoname;
      // TODO(scottmg): Handle kModuleTypeDynamicLoader.
      module.type = ModuleSnapshot::kModuleTypeSharedLibrary;
    }

    std::unique_ptr<ElfImageReader> reader(new ElfImageReader());

    std::unique_ptr<ProcessMemoryRange> process_memory_range(
        new ProcessMemoryRange());
    // TODO(scottmg): Could this be limited range?
    process_memory_range->Initialize(process_memory_.get(), true);
    process_memory_ranges_.push_back(std::move(process_memory_range));

    reader->Initialize(*process_memory_ranges_.back(), base);
    module.reader = reader.get();
    module_readers_.push_back(std::move(reader));
    modules_.push_back(module);

    map = next;
  }
}

void ProcessReaderFuchsia::InitializeThreads() {
  DCHECK(!initialized_threads_);
  DCHECK(threads_.empty());

  initialized_threads_ = true;

  std::vector<zx_koid_t> thread_koids =
      GetChildKoids(*process_, ZX_INFO_PROCESS_THREADS);
  std::vector<zx::thread> thread_handles =
      GetHandlesForThreadKoids(*process_, thread_koids);
  DCHECK_EQ(thread_koids.size(), thread_handles.size());

  for (size_t i = 0; i < thread_handles.size(); ++i) {
    Thread thread;
    thread.id = thread_koids[i];

    if (thread_handles[i].is_valid()) {
      char name[ZX_MAX_NAME_LEN] = {0};
      zx_status_t status =
          thread_handles[i].get_property(ZX_PROP_NAME, &name, sizeof(name));
      if (status != ZX_OK) {
        ZX_LOG(WARNING, status) << "zx_object_get_property ZX_PROP_NAME";
      } else {
        thread.name.assign(name);
      }

      zx_info_thread_t thread_info;
      status = thread_handles[i].get_info(
          ZX_INFO_THREAD, &thread_info, sizeof(thread_info), nullptr, nullptr);
      if (status != ZX_OK) {
        ZX_LOG(WARNING, status) << "zx_object_get_info ZX_INFO_THREAD";
      } else {
        thread.state = thread_info.state;
      }

      zx_thread_state_general_regs_t regs;
      status = thread_handles[i].read_state(
          ZX_THREAD_STATE_GENERAL_REGS, &regs, sizeof(regs));
      if (status != ZX_OK) {
        ZX_LOG(WARNING, status) << "zx_thread_read_state";
      } else {
        thread.general_registers = regs;

        GetStackRegions(regs, memory_map_, &thread.stack_regions);
      }
    }

    threads_.push_back(thread);
  }
}

}  // namespace crashpad
